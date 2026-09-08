from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, CONCISENESS_PROMPT
from dataset import dataset_name
from main import run_weather_agent
from dotenv import load_dotenv

load_dotenv()

judge_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=judge_llm
)

conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    feedback_key="conciseness",
    judge=judge_llm
)


def correctness(outputs: dict, reference_outputs: dict, inputs: dict):
    return correctness_evaluator(
        inputs=inputs["question"],
        outputs=outputs["answer"],
        reference_outputs=reference_outputs["answer"]
    )


def conciseness(outputs: dict, inputs: dict):
    return conciseness_evaluator(
        inputs=inputs["question"],
        outputs=outputs["answer"]
    )


client = Client()

results = client.evaluate(
    run_weather_agent,
    data=dataset_name,
    evaluators=[correctness, conciseness],
    experiment_prefix="weather-agent"
)
