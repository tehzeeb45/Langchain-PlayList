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
prompt = PromptTemplate(
    template = 'Generate 5 interesting fact about {topic}',
    input_variables=['topic']
)
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({'topic':'imran khan'})
print(result)
chain.get_graph().print_ascii()

