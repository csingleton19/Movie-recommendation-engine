
import os
from dotenv import load_dotenv
import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
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

# Load the movies data from the JSON file
with open('movies.json', 'r') as f:
    movies_data = json.load(f)

# Convert the data into a pandas DataFrame
movies_df = pd.DataFrame(movies_data)

# Create a MultiLabelBinarizer for genres and cast
mlb_genre = MultiLabelBinarizer()
mlb_cast = MultiLabelBinarizer()

# Ensure 'director' is a single string value (not a list)
movies_df['director'] = movies_df['director'].apply(lambda x: x[0] if isinstance(x, list) and x else '')

# One-hot encode the 'director', 'title' columns
directors_encoded = pd.get_dummies(movies_df['director'])
titles_encoded = pd.get_dummies(movies_df['title'])

# Apply the binarizer to the genres and cast columns
genres_encoded = mlb_genre.fit_transform(movies_df['genre_ids'])
cast_encoded = mlb_cast.fit_transform(movies_df['cast'])

# Create DataFrames from the encoded data and set the column names
genres_df = pd.DataFrame(genres_encoded, columns=mlb_genre.classes_)
cast_df = pd.DataFrame(cast_encoded, columns=mlb_cast.classes_)

# Concatenate the original DataFrame with the encoded DataFrames
movies_encoded_df = pd.concat([movies_df, genres_df, cast_df, directors_encoded, titles_encoded], axis=1)

# We can drop the original 'genre_ids', 'cast', 'director' columns
movies_encoded_df.drop(['genre_ids', 'cast', 'director'], axis=1, inplace=True)

# Make sure to fill any NA with 0 after one-hot encoding
movies_encoded_df.fillna(0, inplace=True)

# Normalize the vote_average scores
movies_encoded_df['vote_average'] = movies_encoded_df['vote_average'] / 10.0

# Separate vector and non-vector data
non_vector_data = movies_encoded_df[['id', 'original_title', 'overview', 'release_date', 'backdrop_path', 'original_language', 'poster_path', 'title']]
vectors_df = movies_encoded_df.drop(['id', 'original_title', 'overview', 'release_date', 'backdrop_path', 'original_language', 'poster_path', 'title'], axis=1)

# Convert remaining columns to float
vectors_df = vectors_df.astype(float)

# Now let's create the vector database with Pinecone
pinecone.init(api_key=pinecone_api, environment=pinecone_env)  # Initialize Pinecone with your API key

index_name = "movie"  # Name of the Pinecone index

# Delete the index if it already exists
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

# The vectors for the index
vectors = vectors_df.values

# The IDs will be the movie IDs from the original data
ids = non_vector_data['id'].values.astype(str)

# Create a new vector index
pinecone.create_index(name=index_name, dimension=vectors.shape[1], metric="cosine", shards=1)

# Create a Pinecone Index object
index = pinecone.Index(index_name=index_name)

# Upsert the vectors into the index
batch_size = 100  # or adjust this value according to your conditions
for i in range(0, len(vectors), batch_size):
    chunk_ids = ids[i:i + batch_size]
    chunk_vectors = vectors[i:i + batch_size]
    data_to_upsert = list(zip(chunk_ids, chunk_vectors.tolist()))
    index.upsert(vectors=data_to_upsert, batch_size=batch_size)

logging.info("Finished creating the vector database!")

movies_encoded_df.to_pickle("movies_encoded_df.pkl")
vectors_df.to_pickle("vectors_df.pkl")
movies_df.to_pickle("movies_df.pkl")

# logging.info("Finished saving the files!")
# print(movies_encoded_df["Zoë Kravitz"].nunique())
# print(movies_encoded_df["To Catch a Killer"].nunique())
# print(movies_encoded_df)
# for title in movies_df.title:
#     print(title)
