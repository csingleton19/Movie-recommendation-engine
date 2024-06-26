from recommendation_engine import find_similar_movies
from recommendation_engine import find_similar_movies_by_preferences
import pandas as pd
import openai
import json
import os
from dotenv import load_dotenv
import pinecone
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai_api = os.getenv("openai_api")
pinecone_api = os.getenv("pinecone_api")
pinecone_env = os.getenv("pinecone_env")
pinecone_index = os.getenv("pinecone index")
movie_api = os.getenv("tmdb_api")


# Initialize OpenAI API
openai.api_key = openai_api

# Initialize Pinecone Index
index = pinecone.Index(index_name=pinecone_index)

# Logging
logging.info("Finished loading API keys!")



# Load data
movies_df = pd.read_pickle("movies_df.pkl")
movies_encoded_df = pd.read_pickle("movies_encoded_df.pkl")
vectors_df = pd.read_pickle("vectors_df.pkl")

# Check for 'director' and 'actors' in the column names, 
# if they are not in the DataFrame, we should not process them
process_directors = 'director' in movies_df.columns
process_actors = 'actors' in movies_df.columns

# Initialize the unique entities sets
genres = set()
actors = set() if process_actors else None
directors = set() if process_directors else None

# Extract unique genres from the 'genre_ids' column
genres = movies_df['genre_ids'].explode().unique().tolist()

user_preferences = {
    'genres': [],  
    'directors': [],  
    'actors': [],
    'vote_average': 0.65,
    'movie': None
}


def save_user_preferences(user_preferences):
    with open('user_preferences.json', 'w') as f:
        json.dump(user_preferences, f)


def load_user_preferences(filename="user_preferences.pkl"):
    try:
        with open(filename, 'rb') as file:
            user_preferences = pickle.load(file)
    except FileNotFoundError:
        user_preferences = {
            'genres': [],  
            'directors': [],  
            'actors': [],
            'vote_average': 0.65,
            'movie': None
        }
    return user_preferences


def update_user_preferences(user_message, user_preferences):
    # Check for genres
    for genre in genres:
        if genre.lower() in user_message.lower():
            user_preferences.setdefault('genres', []).append(genre)
    
    # Check for directors
    if process_directors:
        for director in directors:
            if director.lower() in user_message.lower():
                user_preferences.setdefault('directors', []).append(director)

    # Check for actors
    if process_actors:
        for actor in actors:
            if actor.lower() in user_message.lower():
                user_preferences.setdefault('actors', []).append(actor)

    # Check for vote average
    if 'vote average' in user_message.lower():
        try:
            # Extract the number after 'vote average' in the user message
            vote_average = float(user_message.split('vote average')[-1].strip())

            # Normalize if the number is between 1 and 10
            if 1 <= vote_average <= 10:
                vote_average /= 10.0

            user_preferences['vote_average'] = vote_average
        except ValueError:
            logging.info("Invalid vote average provided, please provide a valid number.")

    # If no vote average is provided, set a default of 0.65
    if 'vote_average' not in user_preferences:
        user_preferences['vote_average'] = 0.65

    # Check for titles
    if 'movie' in user_message.lower():
        # extract the title from the user message
        title = user_message.split('movie')[-1].strip()
        user_preferences['movie'] = title

    return user_preferences


def handle_user_message(user_message):
    messages = []
    messages.append({"role": "user", "content": user_message})

    # Update user preferences based on user's input
    update_user_preferences(user_message, user_preferences)

    # Save updated user preferences
    save_user_preferences(user_preferences)

    # Initialize assistant_message
    assistant_message = ''

    # Check if user input triggers recommendation engine
    if "recommend" in user_message:
        recommendations = []
        if user_preferences.get('movie'):  # if a specific movie title has been specified
            recommendations = find_similar_movies(user_preferences.get('movie'), movies_df, vectors_df, index, n=5)
        elif any(item in user_message for item in ['genre', 'director', 'actor']):  # if any specified feature is in the user message
            recommendations = find_similar_movies_by_preferences(user_preferences, movies_encoded_df, vectors_df, index, n=10)

        # Convert the recommendations to strings and add to the assistant message
        if recommendations:
            assistant_message = "\nHere are some recommendations: " + ", ".join(recommendations)
        else:
            assistant_message = "I couldn't find any movie that matches your preferences. You may want to adjust your preferences and try again."
    else:
        # Generate a response from the assistant
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=messages,
        )
        assistant_message = response['choices'][0]['message']['content']

    messages.append({"role": "assistant", "content": assistant_message})

    return assistant_message


# print(user_preferences)





