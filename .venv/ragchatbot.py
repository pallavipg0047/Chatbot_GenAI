from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pdfplumber
import streamlit as st # type: ignore
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.header("My First Chatbot")
with st.sidebar:
    st.title("Your document")
    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
#Extract Content from PDF and in chucks
if file is not None:
    with pdfplumber.open(file) as pdf:
        #extract text from it
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    st.write(text)
    
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    #st.write(chunks)
    #get user query and find relevant chunks
    user_question = st.text_input("Type your question here")
    
    #generating embeddings for the chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    
    #vector store for the chunks and their embeddings
    vector_store = FAISS.from_texts(chunks, embeddings)
    
  
    
    #generate answer using the relevant chunks
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={"k": 4})
    
    #define LLM and prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3,max_tokens=500, openai_api_key=OPENAI_API_KEY)
    
    #prompt template for the LLM - its the instruction the defines how the LLM will give the result to end user 
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    
    chain =(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        |llm
        |StrOutputParser(keep_whitespace=True)
        )
    
    if user_question:
        response =chain.invoke(user_question)
        st.write(response)