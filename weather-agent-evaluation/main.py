import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")
agent_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")


@tool
def getWeather(city: str):
    """
    Get weather from openweathermap based on city name
    """
    import requests
    response = requests.get('https://api.openweathermap.org/data/2.5/weather', {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    })
    if response.status_code != 200:
        return "Unable to fetch data from API"
    return response.json()


DB_NAME = "conversation.db"


def run_weather_agent(inputs: dict) -> dict:
    with SqliteSaver.from_conn_string(DB_NAME) as checkpointer:
        checkpointer.setup()
        agent = create_agent(
            model=agent_llm,
            system_prompt="You are a weather agent who answers questions about current weather information by calling tools. Do not respond to any other questions other than weather related queries. Keep your response with a bit of humour and emojis. If you don't know the answer politely say you don't know",
            tools=[getWeather],
            checkpointer=checkpointer
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": inputs["question"]}]},
            config={"configurable": {"thread_id": "eval-thread"}}
        )
        return {"answer": result["messages"][-1].content}
