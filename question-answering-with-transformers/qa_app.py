import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Question Answering", page_icon="🔎")
st.title("🔎 Question Answering with Transformers")
st.write("Paste a passage, ask a question about it, and get the extracted answer.")


@st.cache_resource
def load_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


qa_pipeline = load_pipeline()

context = st.text_area(
    "Context / Passage",
    height=200,
    placeholder="Paste a paragraph of text here...",
)
question = st.text_input("Question", placeholder="Ask something about the passage above...")

if st.button("Get Answer") and context.strip() and question.strip():
    with st.spinner("Thinking..."):
        result = qa_pipeline(question=question, context=context)
    st.success(f"**Answer:** {result['answer']}")
    st.caption(f"Confidence: {result['score']:.3f}")
