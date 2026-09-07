from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

prompt = ChatPromptTemplate(
    [
        ("system",
         "you rewrite chat messages in a {tone} tone for a {audience} audience. Keep the core meaning intact. Respond with ONLY the rewritten chat message - no options, no headers, no explanations, no markdown"),

        ("human", "hey can u send me the report by tmrw, kinda urgent"),
        ("ai", "Hi, could you please send over the report by tomorrow? It's fairly urgent — thanks in advance."),

        ("human", "not gonna make the call today, smth came up"),
        ("ai", "I won't be able to make today's call — something has come up. Apologies for the short notice."),

        ("human", "{message}")
    ]
)

chain = prompt | llm

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = chain.invoke({
        "tone": "professional",
        "audience": "external client",
        "message": user_input
    })
    print(f"Rephrased Message: {response.text}")
