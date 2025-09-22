from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
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
prompt1 = PromptTemplate(
    template = 'Generate a tweet about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template = 'Generate a Linkedin post about  {topic}',
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)
parallel_chain  = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'Explanation' : RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
print(final_chain.invoke({'topic' : 'cricket'}))
