from dotenv import load_dotenv

load_dotenv()
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from openai import OpenAI
import os
import json

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

pc.create_index(
    name="rag",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

data = json.load(open("reviews.json"))

processed_data = []
client = OpenAI()

for review in data["reviews"]:
    response = client.embeddings.create(
        input=review["review"], model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    processed_data.append(
        {
            "values": embedding,
            "id": review["professor"],
            "metadata": {
                "professorName": review["professor"],
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            },
        }
    )


index = pc.Index("rag")
print(processed_data)
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)

print(index.describe_index_stats())
