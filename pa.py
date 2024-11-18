import os
import google.generativeai as gemini
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Authenticate with Gemini API
gemini.configure(api_key=GEMINI_API_KEY)

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to index document with Gemini
def index_document_with_gemini(text):
    # Assumes that Gemini's RAG feature or a similar document indexing API exists
    # Here we pass the document text to Gemini’s RAG model for indexing
    response = gemini.index_document(text=text)
    return response

# Function to perform search with Gemini
def search_with_gemini(query):
    # Perform the search by querying over the indexed document
    response = gemini.search(query=query)
    results = response.get("results", [])
    return results

# Main function
def main():
    # Step 1: Upload document
    file_path = input("Enter the path to the PDF file: ")
    if not os.path.exists(file_path):
        print("File does not exist. Please provide a valid path.")
        return

    print("Processing and indexing the document...")
    text = extract_text_from_pdf(file_path)
    index_response = index_document_with_gemini(text)

    if not index_response:
        print("Failed to index document with Gemini.")
        return

    # Step 2: Handle user queries
    print("The system is ready! Type your query below:")
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Step 3: Retrieve from the document using Gemini search
        print("Searching the document with Gemini...")
        response = search_with_gemini(query)
        if response:
            print("\nAnswer from the document:")
            for result in response:
                print(result["text"])
        else:
            print("\nNo relevant information found in the document.")

if __name__ == "__main__":
    main()
