import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
#from langchain.llms import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_vectorstore(text_chunks):
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = load_embeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
def get_conversation_chain(vectorstore):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=512,
        temperature=0.5,
        do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":books:", layout="centered")
    st.header("Chat with multiple PDFs :books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    user_question=st.text_input("ask a question")
    if user_question:
        handle_user_input(user_question)
    #st.write(user_template.replace("{{MSG}}", "Hello, I am a user."), unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", "Hello, I am a bot.") , unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("your PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here",accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

                
    #st.session_state.conversation


if __name__ == "__main__":
    main()