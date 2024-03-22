
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import pinecone
import numpy as np
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

pinecone.init(api_key = pinecone_api, environment = pinecone_env)
index = pinecone.Index(index_name = pinecone_index)

# # Add this line to list all available indexes
# logging.info(pinecone.list_indexes())

# # Or add this line to describe your specific index
# logging.info(index.describe_index_stats())

movies_df = pd.read_pickle("movies_df.pkl")
movies_encoded_df = pd.read_pickle("movies_encoded_df.pkl")
vectors_df = pd.read_pickle("vectors_df.pkl")


from pinecone.core.exceptions import PineconeProtocolError

def find_similar_movies(movie_title, movies_df, vectors_df, index, n=5):
    # Check if the movie is in the DataFrame
    if movie_title in movies_df['title'].values:
        # If it is, get the vector for the movie
        movie_vector = vectors_df[movies_df['title'] == movie_title].values[0]

        try:
            # Query the Pinecone index for similar movies
            result = index.query([movie_vector.tolist()], top_k=n)
            
            # Get the IDs of the similar movies
            similar_movie_ids = result.ids[0]
            # logging.info(f"Similar movie IDs: {similar_movie_ids}")

            
            # Map the IDs back to movie titles
            similar_movie_titles = movies_df[movies_df['id'].isin(similar_movie_ids)]['title'].values
            # logging.info(f"Similar movie titles: {similar_movie_titles}")
            
            return similar_movie_titles
        except Exception:
            # logging.info("Pinecone query failed, falling back to OpenAI API...")
            return []
    else:
        return movie_title

# movie_title = "John Wick"
# similar_movies = find_similar_movies(movie_title, movies_df, vectors_df, index, n=10)
# print(similar_movies)


def find_similar_movies_by_preferences(user_preferences, movie_encoded_df, vectors_df, index, n=10):
    # Start with all movies
    filtered_df = movie_encoded_df

    # Filter by genres, directors, and actors
    for category in ['genres', 'directors', 'actors']:
        if user_preferences[category]:
            condition = np.logical_or.reduce([filtered_df[feature] == 1 for feature in user_preferences[category]])
            filtered_df = filtered_df[condition]

    # Filter by voting average if provided
    if user_preferences['voting_average'] is not None:
        condition = filtered_df['vote_average'] >= user_preferences['voting_average']
        filtered_df = filtered_df[condition]

    # Filter by movie if provided
    if user_preferences['movie'] is not None:
        condition = filtered_df['title'] == user_preferences['movie']
        filtered_df = filtered_df[condition]

    if len(filtered_df) > n:
        # Convert the filtered DataFrame to vectors
        filtered_vectors = filtered_df.values.tolist()

        try:
            # Query the Pinecone index
            result = index.query(filtered_vectors, top_k=n)

            # Get the IDs of the top N most similar movies
            similar_ids = result.ids

            # Return the titles of the top N most similar movies
            similar_movies = movie_encoded_df.loc[similar_ids]['title']

            return similar_movies
        except Exception:
            logging.info("Pinecone query failed, falling back to OpenAI API...")
            return []
    else:
        return user_preferences



# logging.info(type(pinecone_index))


# ## Test for find_similar_movies_by_preference

# def test_find_similar_movies_by_preferences():
#     # Define a dictionary for user preferences
#     user_preferences = {
#         'genres': ['Action'],
#         'directors': [],
#         'actors': [],
#         'voting_average': None,
#         'movie': None
#     }

#     # Use the function to find similar movies
#     similar_movies = find_similar_movies_by_preferences(user_preferences, movies_encoded_df, vectors_df, index, n=10)

#     # If similar_movies is not None, check if it contains at least 5 movies and logging.info them
#     if similar_movies is not None:
#         assert len(similar_movies) >= 10, "The result should contain at least 5 movies."
        
#         # print the titles of the similar movies
#         for movie in similar_movies:
#             print(movie)
#     else:
#         print("Not enough movies match the given preferences.")




# test_find_similar_movies_by_preferences()
