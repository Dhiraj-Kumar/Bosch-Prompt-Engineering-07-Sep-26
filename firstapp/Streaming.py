from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

conversation_history = [
    SystemMessage(content="You are an AI assistant who gives answers with a bit of humour. Keep your tone professional while answering the questions. Add emojis to make your responses looks nice.")
]

while True:
    user_input = input("You: ")
    conversation_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    response = ''
    for chunk in llm.stream(conversation_history):
        print(chunk.text, end="", flush=True)
        response = response+chunk.text
    conversation_history.append(AIMessage(content=response))
