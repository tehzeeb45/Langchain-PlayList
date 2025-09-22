from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.schema.runnable import RunnableSequence, RunnableParallel
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
    template = 'Generate a tweet about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template = 'Generate a Linkedin post about  {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model, parser),
    'linkedin' : RunnableSequence(prompt2, model, parser)
})
result = parallel_chain.invoke({'topic':'Ai'})
print(result)