import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_EMBEDDING_API_KEY"),
    base_url=os.getenv("NVIDIA_EMBEDDING_BASE_URL")
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


def generate_embedding(text: str, input_type: str = "passage"):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        extra_body={
            "input_type": input_type
        }
    )

    return response.data[0].embedding