from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
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
parser = StrOutputParser()
prompt = PromptTemplate(
    template = 'Write a summary about the following poem- \n {poem}',
    input_variables=['poem']
)


loader = TextLoader('cricket.txt')

docs = loader.load()

print(docs)
print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0])
chain = prompt | model | parser
print(chain.invoke({'poem' : docs[0].page_content}))