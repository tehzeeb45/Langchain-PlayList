import streamlit as st
import os
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate

# ✅ Use your Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.header("Research Tools")

paper_input = st.selectbox("Select Research Paper Name", [
    "Attention Is All You Need", 
    "BERT: Pre-training of Deep Bidirectional Transformers", 
    "GPT-3: Language Models are Few-Shot Learners", 
    "Diffusion Models Beat GANs on Image Synthesis"
])

style_input = st.selectbox("Select Explanation Style", [
    "Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"
])

length_input = st.selectbox("Select Explanation Length", [
    "Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"
])

template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
   - Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
    input_variables=["paper_input", "style_input", "length_input"]
)

# ✅ Use a valid, public model from Hugging Face
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # This model works via API
    model_kwargs={"temperature": 0.7, "max_new_tokens": 300},
     task="text2text-generation"
)

if st.button("Summarize"):
    chain = template | llm
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)