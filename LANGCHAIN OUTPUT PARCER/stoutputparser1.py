from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
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
    template = 'write a detailed report on {topic}',
    input_variables=['topic']
)
# 2st -> summary
template2 = PromptTemplate(
    template = 'Write a 5 line summary on the following text. /n {text}' ,
    input_variables=['text']
)
parser  = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic': 'black hole'})
print(result)