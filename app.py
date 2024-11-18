import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google API key for Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to read PDF content
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create and save vector store
def get_vector_store(text_chunks, db_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(db_name)


# Function to create conversational chain with updated prompt
def get_conversational_chain():
    prompt_template = """
    You are tasked with retrieving relevant research studies or clinical trials based on the context provided. You will generate a Boolean query 
    to retrieve the most relevant studies or trials that are related to the topic mentioned. If no related studies or trials are found in the context, 
    respond with "No relevant studies or clinical trials found in the context."

    Context:\n {context}\n
    Boolean Query for Relevant Studies or Clinical Trials: \n{question}\n

    Answer (with relevant data, if available):
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Function to handle user input and response
def user_input(user_question, db_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print("Reply: ", response["output_text"])


# Main function to handle PDF processing and user interaction
def main():
    print("Chat PDF - Law Bot")

    # Role selection
    role = input("Select Role (AI Engineer / GenAI Engineer): ")
    db_name = f"faiss_index_{role.lower()}"

    # User question input
    user_question = input(f"Ask a Question related to research studies or clinical trials from the {role} PDF Files: ")

    if user_question:
        user_input(user_question, db_name)

    # PDF upload and processing
    pdf_docs = []
    num_files = int(input(f"How many {role} PDFs would you like to upload? "))

    for i in range(num_files):
        file_path = input(f"Enter the path of PDF file {i + 1}: ")
        pdf_docs.append(file_path)

    if pdf_docs:
        print(f"Processing {role} PDFs...")
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks, db_name)
        print(f"{role} PDFs Processed Successfully")


# Run the main function
if __name__ == "__main__":
    main()
