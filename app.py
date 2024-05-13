import streamlit as st
from dotenv import load_dotenv 
from  PyPDF2  import  PdfReader 
from langchain.text_splitter import CharacterTextSplitter  
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains import  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from htmltemplate import css , bot_template , user_template 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os





def get_pdf_text(uploaded_files):
  text  =""
  for pdf in uploaded_files:
       pdf_reader= PdfReader(pdf)
       for page in pdf_reader.pages:
            text+= page.extract_text( )
            
  return text           

def get_text_chunks(raw_text):
     text_splitter = CharacterTextSplitter(
         separator="\n",
         chunk_size= 1000,
         chunk_overlap = 200 ,
         length_function = len 
     )
     chunks = text_splitter.split_text(raw_text)
     return chunks
 
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("file_index")
    # st.write("vector_store entered")  
    return vector_store

# def get_conversational_chain(vector_store):
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vector_store.as_retriever(),
#         memory=memory
#     )
#     st.write("entered!!!!")  
#     return conversation_chain
def get_conversational_chain():
    prompt_template = """
        analyse the file and answer the question 
        List of directories and files: \n {context}?\n
        Question: {question}\n
        Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def handle_user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("file_index", embeddings=embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    chain = get_conversational_chain()
    response= chain({"input_documents": docs, "question": question}, return_only_outputs=True)     
    # st.write(response)
    if response :
      output=response["output_text"]
      st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
      st.write(bot_template.replace("{{MSG}}", output), unsafe_allow_html=True)
    
def main():
    
    
    # Header
    load_dotenv()

    st.title("Chat with multiple PDFs : books:")
    st.write(css, unsafe_allow_html = True)
   
    if  "conversation" not in  st.session_state :
        st.session_state.conversation = None
    # Text input
    if  "responses" not in  st.session_state :
     st.session_state.response = None
    question = st.text_input("Ask a question for the PDFs uploaded")
    if question :
        handle_user_input(question)
    
    
    

    # Sidebar
    st.sidebar.subheader("Your Documents")
    uploaded_files = st.sidebar.file_uploader("Upload your PDFs", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
    # Process button
     if st.sidebar.button("Analyse"):
        with  st.spinner(f"Processing"):
            # get text 
            raw_text = get_pdf_text( uploaded_files)
            # st.write(raw_text)
            # get text chunks
            text_chunks = get_text_chunks(raw_text) 
            # st.write(text_chunks)
            get_vector_store(text_chunks)
            # You can add your processing logic here
            
            
            
        

if __name__ == "__main__":
    main()
