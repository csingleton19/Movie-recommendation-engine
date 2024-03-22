import os
import json
import pandas as pd
import requests
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

# Number of pages to fetch from the API
num_pages = 10  # Reduce this while testing

all_movies = []
# Loop over the page numbers
for page in range(1, num_pages + 1):
    # The Movie Database (TMDB) API URL for popular movies
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={movie_api}&language=en-US&page={page}"

    # Make a request to the TMDB API
    response = requests.get(url)

    # Check if the response contains 'results'
    if 'results' in response.json():
        # Extract the 'results' field from the response JSON (list of popular movies)
        movies = response.json()['results']

        for movie in movies:
            # Make a request to the TMDB API for the credits of the current movie
            credits_url = f"https://api.themoviedb.org/3/movie/{movie['id']}/credits?api_key={movie_api}"
            credits_response = requests.get(credits_url)

            if 'cast' in credits_response.json():
                # Extract the top 5 cast members
                movie['cast'] = [actor['name'] for actor in credits_response.json()['cast'][:5]]
            if 'crew' in credits_response.json():
                # Extract the director
                movie['director'] = [member['name'] for member in credits_response.json()['crew'] if member['job'] == 'Director']

            all_movies.append(movie)

    else:
        logging.info(f"No results found for page {page}.")
        break  # Exit the loop

logging.info("Finished Loading movie results!")

logging.info("Making the genre column human readable")

# Get the genre list from TMDB API
genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={movie_api}&language=en-US"
genre_response = requests.get(genre_url)
genre_response_json = genre_response.json()

# Check if 'genres' key exists in the response
if 'genres' in genre_response_json:
    genre_list = genre_response_json['genres']

    # Create a dictionary to map genre IDs to genre names
    genre_dict = {genre['id']: genre['name'] for genre in genre_list}

    # Replace genre IDs with genre names in the movies data
    for movie in all_movies:
        movie['genre_ids'] = [genre_dict[genre_id] for genre_id in movie['genre_ids'] if genre_id in genre_dict]
else:
    logging.info(f"Key 'genres' not found in the genre response. Full response: {genre_response_json}")


logging.info("Starting results dump")

with open('movies.json', 'w') as f:
    json.dump(all_movies, f)



