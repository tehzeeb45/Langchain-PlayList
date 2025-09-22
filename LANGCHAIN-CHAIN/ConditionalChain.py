from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal
import os

# .env file load karega
load_dotenv()

# Model init
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

# ------------------ Pydantic schema ------------------
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description="Give the sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# ------------------ Classifier Prompt ------------------
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback "
        "strictly as JSON with the field 'sentiment'.\n\n"
        "Feedback: {feedback}\n\n"
        "{format_instructions}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser2

# ------------------ Response Prompts ------------------
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment"),
)

# ------------------ Full Chain ------------------
chain = classifier_chain | branch_chain

# ------------------ Run ------------------
print(chain.invoke({"feedback": "This is a beautiful phone"}))
