import subprocess
import streamlit as st

scripts = ["recommendation_engine.py", "chatbot.py"]

for script in scripts:
    subprocess.call(['python', script])

from chatbot import handle_user_message



st.title('Movie Recommendation Chatbot')


st.write("Suggested formats for best results:")
st.write("Recommend horror movies")
st.write("Recommend movies like X-Men Origins")
st.write("Recommend movies with Jack Nicholson")
st.write("Recommend is the key word to use for best results! Just be sure to press send instead of hitting enter.")
user_input = st.text_input("Ask the chatbot something and press send")
if st.button('Send'):
    response = handle_user_message(user_input)
    st.write(response)

