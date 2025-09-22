import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
os.environ["HF_TOKEN"] = "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"
# ============ Streamlit UI ============
st.title("🎥 YouTube Transcript Q&A (RAG)")

video_url = st.text_input("Enter YouTube Video URL:")
user_question = st.text_area("Enter your question:")

if st.button("Get Answer"):
    try:
        # ---- Extract video_id ----
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        else:
            video_id = video_url.strip()

        # ---- Transcript ----
        yt = YouTubeTranscriptApi()
        transcript_list = yt.fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)

        # ---- Split into chunks ----
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        # ---- Embeddings + FAISS ----
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectore_Store = FAISS.from_documents(chunks, embedding_model)
        retriever = vectore_Store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

        # ---- HuggingFace Model ----
        # 🔑 Set Hugging Face Token


        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",   # base model is open, no token required
            max_new_tokens=300,
            temperature=0.3,
             token=os.environ["HF_TOKEN"]
        )
        llm = HuggingFacePipeline(pipeline=generator)

        # ---- Prompt ----
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            Explain your answer in detail, step by step.
            Do not answer just yes/no.

            {context}
            Question: {question}
            """,
            input_variables=['context', 'question']
        )

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        # ---- Chain ----
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

        # ---- Answer ----
        if user_question:
            answer = main_chain.invoke(user_question)
            st.subheader("📖 Answer")
            st.write(answer)
        else:
            st.warning("⚠️ Please enter a question.")

    except TranscriptsDisabled:
        st.error("❌ No captions available for this video.")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
