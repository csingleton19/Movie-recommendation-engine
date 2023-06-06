import subprocess
from chatbot import handle_user_message
import streamlit as st

scripts = ["data_collection.py", "vector_database.py", "recommendation_engine.py", "chatbot.py"]

for script in scripts:
    subprocess.call(['python', script])



st.title('Movie Recommendation Chatbot')


st.write("Suggested formats for best results:")
st.write("Please recommend horror movies")
st.write("Please recommend movies like X-Men Origins")
st.write("Please recommend movies with Jack Nicholson")
st.write("Recommend is the key word to use for best results!")
user_input = st.text_input("Ask the chatbot something")
if st.button('Send'):
    response = handle_user_message(user_input)
    st.write(response)

