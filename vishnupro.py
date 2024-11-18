import os
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq  # Import Groq library
import serpapi

# Load environment variables from .env file
load_dotenv()

# Retrieve the API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Groq API key
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")  # SerpApi key (stored in the .env file)

# Streamlit UI Setup
st.title("AI Information Retrieval Dashboard")

# Step 1: File Upload and Column Selection
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())
    main_column = st.selectbox("Select main column for search entities", df.columns)

    # Step 2: Dynamic Query Input
    custom_prompt = st.text_input("Define your search prompt using {entity} placeholder", "Get me the email address of {entity}")

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

    # Function to Extract Information Using Groq API Client
    def extract_info(search_results):
        # Initialize the Groq client using the API key
        client = Groq(api_key=GROQ_API_KEY)
        
        # Construct the prompt to pass to Groq for extraction
        prompt = f"Extract the required information from the following results: {str(search_results)}"
        
        try:
            # Use the `parse()` method (correct method in Groq API)
            response = client.parse(prompt)
            
            # Check if the response contains relevant data
            if 'data' in response:
                return response['data']  # Return extracted data
            else:
                st.warning("No relevant data found in Groq API response.")
                return "No relevant data"
        except Exception as e:
            st.error(f"Error with Groq API request: {e}")
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
