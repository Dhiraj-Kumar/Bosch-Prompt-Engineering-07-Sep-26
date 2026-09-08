from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

client = Client()

dataset_name = "weather-agent-dataset"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[
            {"question": "What's the weather in Pune?"},
            {"question": "Tell me the weather in London today"},
            {"question": "Who won the cricket match yesterday?"},
        ],
        outputs=[
            {"answer": "Current weather in Pune with temperature and conditions"},
            {"answer": "Current weather in London with temperature and conditions"},
            {"answer": "A polite refusal, since this is not a weather query"},
        ],
        dataset_id=dataset.id
    )
