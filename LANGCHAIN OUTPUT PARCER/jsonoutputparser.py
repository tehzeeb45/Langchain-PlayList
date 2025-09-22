from dotenv import load_dotenv
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import JsonOutputParser


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
parser =  JsonOutputParser()
template = PromptTemplate(
    template = 'Give ma the name , age  and city of a fictional personal \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instruciton()}
) 
chain = template | model | parser
result = chain.invoke()
print(result)