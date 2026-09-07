from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

examples = [
    {
        "ticket": "My invoice shows a charge I don't recognize from last month.",
        "output": "CATEGORY: billing \n PRIORITY: P2 \n TEAM: finance-ops \n SUMMARY: Unrecognized charge on invoice",
    },
    {
        "ticket": "The app crashes every time I try to upload a photo larger than 5MB.",
        "output": "CATEGORY: bug \n PRIORITY: P1 \n TEAM: mobile-eng \n SUMMARY: Crash on photo upload >5MB",
    },
    {
        "ticket": "Can you add dark mode to the settings page?",
        "output": "CATEGORY: feature-request \n PRIORITY: P4 \n TEAM: product \n SUMMARY: Request for dark mode",
    },
]

example_prompt = PromptTemplate(
    template="Ticket {ticket} \n {output}",
    input_variables=["ticket", "output"]
)

fewshot_template = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    prefix="You are a ticket triage assistant for our internal system. Classify each ticket using our exact schema: CATEGORY, PRIORITY (P1-P4), TEAM, SUMMARY. TEAM must be one of: finance-ops, mobile-eng, backend-eng, product, security.",
    suffix="Ticket: {ticket}",
    input_variables=["ticket"]
)

chain = fewshot_template | llm | StrOutputParser()

print(chain.invoke(
    {"ticket": "Users are seeing other people's account data on login."}))
