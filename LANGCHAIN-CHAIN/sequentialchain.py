from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
# .env file load karega
load_dotenv()

# token read karega environment se
 ## hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
 ## print("Loaded Token:", hf_token)  # check token load ho raha hai ya nahi

llm  = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
   ## huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)
prompt1 = PromptTemplate(
    template = 'Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template = 'Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)
parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic': 'Unemployment in Pakistan'})
print(result)
chain.get_graph().print_ascii()
