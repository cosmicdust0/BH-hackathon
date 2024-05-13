import sys
sys.path.append("/home/pranav/pranav/PRISM/TEXT_SUMMARIZATION_PRISM/code/webpageScrape/webchat/lib/python3.10/site-packages")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

def get_text_from_file(file_content):
    # No need to read from file, already received the content
    text = file_content
    print("Text: ", text)
    return text

def get_text_chunks(text, chunk_size=10000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("file_index")
    print("HOHOHOH")

def get_conversational_chain():
    prompt_template = """
        You are a bot which performs the following objectives strictly:
        Objectives:
        1. You are given a list of directories and their containing directories and files up to a depth of 3.
        2. You are also given a question asking to find the location of a given directory or a file in abstract and natural language.
        3. Your job is to find the most suitable location of the directory or a file in the given directories.
        4. If and only if there are multiple suitable locations, you should choose the top 3 which is closest to the given question.
        5. If the file or directory is not found, you should return "File not found".
        List of directories and files: \n {context}?\n
        Question: {question}\n

        Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.json
    user_question = data['question']
    file_content = data['file_content']
    flag = data['flag']

    print("FLAG: ", flag)
    print("FILE CONTENT: ", file_content)

    if flag == "1":
        with open("output.txt", 'w') as file:
            file.write(file_content)
        text = get_text_from_file(file_content)
        text_chunks = get_text_chunks(text)
        get_vector_store(text_chunks)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        new_db = FAISS.load_local("file_index", embeddings=embeddings)
        docs = new_db.similarity_search(user_question)
        print("DOCS: ", docs)

        chain = get_conversational_chain()

        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        return jsonify({"response": response["output_text"]})
    else:
        with open("output.txt", 'r') as file:
            file_content = file.read()
        text = get_text_from_file(file_content)
        text_chunks = get_text_chunks(text)
        get_vector_store(text_chunks)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("file_index", embeddings=embeddings)
        docs = new_db.similarity_search(user_question)
        print("DOCS: ", docs)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return jsonify({"response": response["output_text"]})
    

if __name__ == "__main__":
    app.run()
