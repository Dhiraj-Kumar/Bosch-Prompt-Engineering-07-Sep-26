from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

prompt = load_prompt('travel_prompt.json')

chain = prompt | llm | StrOutputParser()

for chunk in chain.stream({'username': 'Dhiraj', 'destination': 'Singapore', 'start_date': '25-01-2026', 'end_date': '30-01-2026', 'interests': 'Adventure', 'travel_style': 'Solo', 'dietary_preferences': 'Non-Vegetarian', 'budget': '100000'}):
    print(chunk, end="", flush=True)
