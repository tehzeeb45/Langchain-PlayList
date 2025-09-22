from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.schema.runnable import RunnableSequence

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

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template = 'Explain the following joke {text}',
    input_variables=['text']
)
parser = StrOutputParser()
chain = RunnableSequence(prompt1, model,parser, prompt2, model, parser)
print(chain.invoke({'topic' : 'AI'}))

