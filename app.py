import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
def get_pdf_text(pdf_docs):
    '''
    Return a string of all the text in uploaded PDFs
    '''
    text=""
    for pdf in pdf_docs: #Loops through each file
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages: #Loops through each page
            text += page.extract_text()
    return text
def get_text_chunks(text):
    '''
    Divide the string into chunks, paragraphs
    '''
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, #For the case when the paragraph is splitted in the middle, the overlap will take the last 200 characters of the previous chunk
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
def get_vectorstore(text_chunks):
    '''
    Create vector store (Embedding)
    '''
    embeddings = HuggingFaceInstructEmbeddings(model_name="srikanthmalla/hkunlp-instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) #Will be stored in our machine
    return vectorstore
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperature":0.5,"max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDFs Chat", page_icon=":shark:")
    st.write(css, unsafe_allow_html=True)
    #Allow us to use st.session_state.conversation in the main 
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Multiple PDFs Chat")
    user_question = st.text_input("Enter a question about your documents!")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs=st.file_uploader("Upload your documents here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #Get the PDFs text
                raw_text = get_pdf_text(pdf_docs)
                #Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #Create vector store (Embedding using Instructor)
                vectorstore = get_vectorstore(text_chunks)
                #Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
if __name__ == '__main__':
    main()