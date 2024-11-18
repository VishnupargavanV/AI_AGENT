import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import serpapi

# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Gemini API key
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")  # SerpApi key (stored in the .env file)

# Initialize the Google Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Streamlit UI Setup
st.title("AI Information Retrieval Dashboard")

# Step 1: File Upload and Column Selection
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())
    main_column = st.selectbox("Select main column for search entities", df.columns)

    # Step 2: Dynamic Query Input
    custom_prompt = st.text_input("Define your search prompt using {entity} placeholder", "Find information about {entity}")

    # Function to Perform Web Search using SerpApi
    def search_web(entity):
        search_query = custom_prompt.replace("{entity}", entity)
        
        # Initialize SerpApi client
        client = serpapi.Client(api_key=SEARCH_API_KEY)
        
        try:
            # Perform the search query (corrected format for query)
            results = client.search({
                'engine': 'google',
                'q': search_query
            })
            return results
        except Exception as e:
            st.error(f"Error with SerpApi request: {e}")
            return None

    # Function to Extract Information Using Google Gemini API with a Simple Prompt Template
    def extract_info(search_results):
        # Construct the prompt for Gemini
        prompt = f"Extract the relevant information from the following search results: {str(search_results)}"
        
        try:
            # Use the gemini-pro-vision model to generate content based on the search results
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Generate content using the model
            response = model.generate_content([f"Extract relevant information from the following results: {str(search_results)}"])
            
            # Check the correct structure of the response and get the text content
            generated_info = response.get('text', None)  # Correct way to access the text content

            if generated_info:
                st.write(f"Information: {generated_info}")
                return generated_info
            else:
                st.warning("No relevant data found in Gemini API response.")
                return "No relevant data"
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg:
                st.error("Invalid API Key. Please enter a valid API Key.")
            else:
                st.error(f"Failed to configure API due to {error_msg}")
            return None

    # Step 3: Process Entities and Retrieve Information with Loading Indicator
    results = []
    with st.spinner("Processing..."):
        for entity in df[main_column].unique():
            st.write(f"Searching for: {entity}")  # Display the current entity being processed
            search_results = search_web(entity)
            if search_results:
                parsed_info = extract_info(search_results)
                results.append({"Entity": entity, "Information": parsed_info})
            else:
                results.append({"Entity": entity, "Information": "No data found"})

    # Step 4: Display and Download Results
    results_df = pd.DataFrame(results)
    st.write("Extracted Information", results_df)

    # Download button for CSV results
    st.download_button(label="Download results as CSV", data=results_df.to_csv(index=False), mime="text/csv")
