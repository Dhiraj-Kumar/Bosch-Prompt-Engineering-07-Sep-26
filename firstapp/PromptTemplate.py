from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

template = PromptTemplate(
    template="Write an article in 100 words on the topic of {topic}", input_variables=["topic"])

# prompt1 = template.invoke({'topic': 'AI'})
# prompt2 = template.invoke({'topic': 'Cyber Security'})

# response = llm.invoke(prompt1)

# print(response.text)

chain = template | llm | StrOutputParser()

print(chain.invoke({'topic': 'AI'}))
