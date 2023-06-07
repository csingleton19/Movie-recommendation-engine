# Movie Recommendation Engine

## Index:

 * Purpose
 * Setup and configuration
 * Limitations
 * Disussion
 * Future Work:
 * Methodology (Note, I chose this to go last because of the length of the section, it breaks down why I made the choices I did)

## Purpose:
    
The purpose of this project is to build a simple chatbot for people interested in getting movie recommendations, using The Movie Database (TMDB) as the data source. Using a range of their preferences, the user should be able to get movie recommendations base

## Setup and Configuration:

This project was set up locally using a conda environment that runs Python 3.10.11. The following libraries were installed via conda, and if they were unavailable via conda install, then pip was used. The libraries that are used are:

* os
* json
* requests
* pandas
* numpy 
* pinecone
* sklearn
* pickle
* logging 
* streamlit

There are also functions that were created in one script that were imported into another, but as long as every file on github gets included in the same folder + all dependencies installed, there should be no problems with running this code locally. The one thing that will have to for sure be manually configured is API keys. You will need API keys with the following to run this locally:

* OpenAI API
* Pinecone API
* TMDB API

If you want to use others for your stack, you will need to adjust the code yourself to account for that. Also this was set up on Ubuntu 20.04, so the following commands should work for Unix/Mac - but Windows users that aren't using WSL may need to tweak the code to match Windows commands

For easy install and environment set-up, download the 'config.yaml' file and run the following command:

conda env create -f config.yml

This will automatically set up the conda environment for someone to be able to run it locally. Then use the following command from the terminal to activate the environment:

conda activate cs_movie_rec

cd into the directory where all the files are stored after downloading them:

cd /path/to/folder/for/movie_rec_engine

At this point you can run:

streamlit run run.py

I also have it currently set up so that other people can connect from other machines as opposed to running it locally. To do this, you would just need to add the following line to the end of your streamlit command:

--server.address 72.89.XX.XX:8501, where 72.89.XX.XX is the IP address of the machine that is currently running the program, and 8501 is the default port for connected where it runs on.



They would still need to set up the API keys, but outside of that, as long as the files are in the same parent folder, everything should work!


## Limitations:

* API Limit Rates: Using the free version of a few APIs/a vector database, it was really easy to exceed the limits set by this. Using a paid version would enable higher scalability
* Local hardware: I'm using an older laptop, so this definitely affects processing power + RAM, so I had to put some limits on the programs i.e. I used a small number of pages from the TMDB API relavtive to what I could have used on a newer machine, or on the cloud

## Discussion:

This was a really fun project to work on, it was outside of the scope of some things I usually work on - but definitely fun to learn! I tried following the Twelve-Factor App methodology as best as I could, but some parts were not applicable, i.e. concurrency: I could have set up API calls to work concurrently, however the rate limits and RAM issues prevented me from doing that. I had to scale it back and work in batches instead. There was no need for dev/prod parity as there is a single environment. On the other hand, logging is incorporated, the dependencies are explicitly declared by importing the required modules in the beginning, and declarative statements are used. 

## Future Work:

#### Low Hanging Fruit: 

* Cloud Migration: As mentioned earlier, running this code on an older machine is suboptimal for data collection and processing power. Migrating this project to a cloud would allow for not only improvements in those areas, but make it easily accessible in the form of a web-app (or other)
* Making certain parts reusable: API imports
* Improving the recommendation engine by making it so people can get recommendations based off of multiple movies, or a movie and features they would like to focus on
* Improving the recommendation engine by making it so that it handles errors better. Right now, any error that would normally cause it to fail will get ignored and rely on the OpenAI API recommendation. This was chosen because there were some intermitten issues issues, i.e. the pineconeAPI connection was not always stable, which would cause the entire process to fail - so I felt that relying on the OpenAI API for recommendations was a good temporary stopgap measure to overcome these intermitten issues. In a production environment, I would work on addressing these issues as opposed to sidestepping them (or addressing them as quickly as possible if there were more pressing needs), however I felt for this exercise, this was a trade-off I could justify. 

#### Everything else:

* Setting up API keys as environment variables, as opposed ot importing them via json the way I currently have it set up
* I could improve the testing that I did - my tests right now are rather simple and minimalistic, but enhancing them in the future would be important - especially in the context of productionizing the code


## Methodology

I will go through the scripts and explain what each one does, and if there is something I feel like I need to explain why I chose it, I will do so too

#### data_collection.py

1) The code you provided is a script to fetch movie data from The Movie Database (TMDB) API

2) It retrieves the details of popular movies, their casts, and directors

3) The genres of the movies are also fetched and made human-readable (i.e., genre IDs are replaced with their corresponding genre names).

4) The code also tries to load a JSON file named 'api_key_list.json' which is expected to contain the necessary API keys

5) The keys are used to make requests to the TMDB API, Pinecone API, and OpenAI API (although the Pinecone and OpenAI keys aren't used in this script)

6) The fetched movie data is then stored in a JSON file named 'movies.json'.

#### vector_database.py

1) API Keys Loading: The initial part of the script reads API keys from a JSON file, similar to the previous script. In addition to the API keys, it also loads the pinecone_env variable

2) Loading Movie Data: The script then loads the movie data from a JSON file named 'movies.json' into a Python list. This list is then converted into a pandas DataFrame for easier manipulation

3) Data Preprocessing: The script then pre-processes the DataFrame to prepare it for vectorization

* I used MultiLabelBinarizer is used to one-hot encode the 'genre_ids' and 'cast' columns of the DataFrame (since a movie can belong to multiple genres and have multiple cast members) 

* The 'director' column is ensured to be a single string value, and then the 'director' and 'title' columns are also one-hot encoded. I chose one hot encoding for a few reasons:
    * I chose to use cosine similarity (which I will get to later), and it is good at handling sparse data, which is often the case after one-hot encoding. It only considers non-zero dimensions while ignoring zero dimensions, which makes it computationally efficient - so it seemed like a logical choice
    * Another reason is because cosine similarity cares about the direction of the vector, not the magnitude - so with one-hot encoding, two vectors of similar directions and not magnitudes will rightfully be deemed similar

4) The original DataFrame is then concatenated with these encoded DataFrames to create a new DataFrame with encoded features. The original 'genre_ids', 'cast', 'director' columns are dropped since their encoded versions are now present

5) The 'vote_average' scores are then normalized by dividing by 10, and any NA values are filled with 0

6) Data Splitting: The DataFrame is then split into non-vector data (which includes information like 'id', 'original_title', 'overview', etc.) and vector data (which includes the encoded features and other numerical features)

7) Vector Database Creation: The script then uses Pinecone to create a vector database. This involves initializing Pinecone with the loaded API key and environment, creating a new vector index (deleting any existing index with the same name), and upserting the vectors into the index

8) Data Storing: Finally, the script stores the encoded DataFrame, the vectors DataFrame, and the original movies DataFrame into pickle files for later use

#### recommendation_engine.py

 1)   This script starts by locating a file named api_key_list.json, which is expected to contain various API keys needed for this script, including movies_api, pinecone_api, openai_api, and pinecone_env

2) Pinecone is a vector database service that's used for similarity search. The script initializes Pinecone with the appropriate API key and environment, and creates a Pinecone Index object

3) The script then loads three DataFrames from pickled files: movies_df (which contains information about movies), movies_encoded_df (which contains the same information in a form that's easier to use for machine learning), and vectors_df (which contains vector representations of the movies)

4) The find_similar_movies function takes a movie title, the movies DataFrame, the vectors DataFrame, the Pinecone index, and a number of similar movies to return. It checks if the given movie title exists in the DataFrame, and if it does, gets its vector representation and queries the Pinecone index for similar movies. The function returns the titles of the similar movies

* I considered three main things for the recommendation enging: Cosine Similarity, Manhattan Distance, and Euclidean Distance. 
    * Cosine Similarity: Like I mentioned earlier, this one is strong with dealing with sparse data - and using one-hot encoding ensured there is plenty of sparse data. In the context of a move recommendation system, it is only concerned about the angle between the two vectors, which means it cares more about common features and what those features have in common than the number of times something has been watched. One last strength it has is dealing with high dimensional data, and my dataframes were over a thousand columns x hundreds of rows
    * Euclidean Distance: This one is a distance metric that tends to work well in datasets with lower dimensionality. For two points (x1,y1) and (x2,y2), the Euclidean distance is sqrt((x2-x1)^2 + (y2-y1)^2). It's a straight-line distance between two points - so in two dimensions it is equivalent to the Pythagorean Theorum
    * Manhattan distance: I considered this one as well, but like Euclidean distance, this one is better suited for lower dimensionality. This one would be better suited for a recommendation system that cares about differences in ratings. The Manhattan Distance is the total sum of the difference between the x-coordinates and y-coordinates. For two points (x1,y1) and (x2,y2), the Manhattan distance is |x2-x1| + |y2-y1|. It's called the "Manhattan distance" because it's similar to the block-by-block path a taxi would have to drive in a city like Manhattan.

5) The find_similar_movies_by_preferences function takes a dictionary of user preferences, the movies DataFrame, the vectors DataFrame, the Pinecone index, and a number of similar movies to return. It filters the DataFrame based on the user preferences and converts the filtered DataFrame to vectors. It then queries the Pinecone index for similar movies and returns the titles of the similar movies

6) The user preferences can include preferred genres, directors, and actors, a minimum voting average, and a specific movie title. If the number of movies that match the user preferences is less than the number of similar movies to return, the function simply returns the user preferences

* Note: Both functions include error handling that returns the original input if the Pinecone query fails, so that way the OpenAI API can give recommendations as a secondary measure

#### chatbot.py

1) The script starts off by importing the necessary libraries, functions, and loads API keys for OpenAI and Pinecone from a JSON file. It sets up the API for OpenAI and creates an instance of a Pinecone index (probably used for vector search)

2) It loads three dataframes that are pickled: movies_df, movies_encoded_df, vectors_df. These probably contain information about movies, their encoded features, and their corresponding vector representations

3) This section extracts unique genres, actors, and directors from the dataframes to build recommendation criteria

4) It maintains a dictionary user_preferences to store user's movie preferences, and includes functions to save and load these preferences

5) update_user_preferences(user_message, user_preferences): This function updates the user_preferences dictionary based on the user's input message

6) handle_user_message(user_message): This is the main function for interaction with the user. It processes the user's input, updates preferences, and generates movie recommendations. If the user input triggers the recommendation engine, the function calls find_similar_movies() or find_similar_movies_by_preferences() to provide recommendations

7) If not, it uses OpenAI API to generate a chat response, which could be used for general conversation or specific queries not related to movie recommendation
