from langchain_huggingface import HuggingFaceEndpoint
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=100
)

response = llm.invoke("What is the capital of Pakistan?")
print(response)
