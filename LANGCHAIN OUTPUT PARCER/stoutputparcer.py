from dotenv import load_dotenv
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# .env file load karega
load_dotenv()

# token read karega environment se
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("Loaded Token:", hf_token)  # check token load ho raha hai ya nahi

llm  = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

# 1st -> Detailed report
template1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text.\n{text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'black hole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})
result1 = model.invoke(prompt2)

print(result1.content)
