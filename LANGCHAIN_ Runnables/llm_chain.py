from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
# .env file load karega
load_dotenv()

# token read karega environment se
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("Loaded Token:", hf_token)  # debugging ke liye

# Model init with token
llm  = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token   # yaha token pass karo
)

model = ChatHuggingFace(llm=llm)

# Prompt
prompt = PromptTemplate(
    template='Suggest a catchy blog title about {topic}',
    input_variables=['topic']
)
chain = LLMChain(llm = llm, prompt=prompt)
topic = input('Enter the input')
output = chain.run(topic)
print("Generated Blog title",output)