from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

template = PromptTemplate(
    template="Write an article in 500 words on the topic of {topic}", input_variables=["topic"])

template2 = PromptTemplate(
    template="Create 5 MCQ type questions based on the below article \n {article}")

chain = template | llm | StrOutputParser() | template2 | llm | StrOutputParser()

print(chain.invoke({'topic': 'AI'}))
