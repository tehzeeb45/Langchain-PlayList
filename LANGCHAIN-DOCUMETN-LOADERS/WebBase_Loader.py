from langchain_community.document_loaders import WebBaseLoader
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
    template = 'Answer the following question \n {question} from the following text - \n {poem}',
    input_variables=['question','text']
)

## from bs4 import BeautifulSoup
url = 'https://mis.pwwf.punjab.gov.pk/UserHome.aspx'
loader = WebBaseLoader(url)
docs = loader.load()
chain= prompt | model | parser
print(chain.invoke({'question' : 'what is the talent scholarship?', 'text':docs[0].page_content}))