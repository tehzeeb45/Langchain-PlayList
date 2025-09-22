from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
# .env file load karega
load_dotenv()

def word_count(text):
    return len(text.split())
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
prompt1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template = 'Summarize the following text \n {text}',
    input_variables=['text']
)
report_gen = RunnableSequence(prompt1, model, parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)
final_chain = RunnableSequence(report_gen, branch_chain)
print(final_chain.invoke({'topic' : 'pakistan vs india'}))