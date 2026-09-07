from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title("Travel Planner")
username = st.text_input("Name")
destination = st.text_input("Destination")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
budget = st.slider("Select Budget", 10000, 1000000)
interest = st.multiselect(
    "Interests", ["Adventure", "Beaches", "Nature", "Shopping"])
travel_style = st.multiselect(
    "Travel Style", ["Luxury", "Solo", "Group", "Family"])
dietary_preferences = st.selectbox(
    "Dietary Preferences", ["Vegetarian", "Non-Vegetarian", "Vegan"])

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

prompt = load_prompt('travel_prompt.json')

chain = prompt | llm | StrOutputParser()

if st.button("Submit"):
    st.write_stream(chain.stream({'username': username, 'destination': destination, 'start_date': start_date, 'end_date': end_date,
                    'interests': interest, 'travel_style': travel_style, 'dietary_preferences': dietary_preferences, 'budget': budget}))
