from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
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
    template = 'Write a joke  about {topic}',
    input_variables=['topic']
)
joke_gen = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'work_count' : RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen, parallel_chain)
print(final_chain.invoke({'topic': 'AI'}))