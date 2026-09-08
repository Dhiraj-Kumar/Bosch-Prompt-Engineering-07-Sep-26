from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.tools import tool
from dotenv import load_dotenv
import os
import requests
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

API_KEY = os.getenv("WEATHER_API_KEY")


@tool
def getWeather(city: str):
    """
    Get weather from openweathermap based on city name
    """
    response = requests.get('https://api.openweathermap.org/data/2.5/weather', {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    })

    if response.status_code != 200:
        return "Unable to fetch data from API"

    return response.json()


DB_NAME = "conversation.db"

with SqliteSaver.from_conn_string(DB_NAME) as sqlcheckpointer:
    sqlcheckpointer.setup()
    agent = create_agent(
        model=llm,
        system_prompt="You are a weather agent who answers questions about current weather information byt calling tools. Do not respond to any other questions other than weather related queries. Keep your response with a bit of humour and emojis. If you don't know the answer politely say you don't know",
        tools=[getWeather],
        checkpointer=sqlcheckpointer
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        result = agent.invoke({"messages": [
            {"role": "user", "content": user_input}
        ]}, config={"configurable": {"thread_id": "thread1"}})
        print(result["messages"][-1].text)
