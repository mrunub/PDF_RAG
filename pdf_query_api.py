import warnings
import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Setup the embeddings and the retriever outside the route to reuse across requests
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the Azure OpenAI Chat model
model_azure = AzureChatOpenAI(
    azure_deployment="",
    openai_api_version="",
    model="",
    openai_api_key='',
    azure_endpoint=""
)

def process_pdf_and_prepare_retriever(pdf_file):
    # Load PDF, split text, and create retriever
    loader = PyPDFLoader(pdf_file)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)
    
    vectorstore = FAISS.from_documents(texts, embedding=embedder)
    retriever = vectorstore.as_retriever()
    retriever.search_kwargs["distance_metric"] = 'cos'
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20
    
    return retriever

@app.route('/query', methods=['POST'])
def query_pdf():
    # Endpoint to handle PDF query
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
    
    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "No PDF file selected"}), 400
    
    if pdf_file and allowed_file(pdf_file.filename):
        # Process PDF and prepare retriever
        retriever = process_pdf_and_prepare_retriever(pdf_file)
        
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Use Azure Chat model for Conversational Retrieval Chain
        model = model_azure
        qa_chain = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
        
        chat_history = []
        result = qa_chain({"question": question, "chat_history": chat_history})
        
        return jsonify(result)
    else:
        return jsonify({"error": "Invalid file type, only PDF files are allowed"}), 400

def allowed_file(filename):
    # Check if file type is allowed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

if __name__ == '__main__':
    app.run(debug=True)
