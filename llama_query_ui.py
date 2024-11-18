# aws_query_interface.py
import os
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import PromptTemplate, StorageContext, load_index_from_storage
from llamacloud_init import openai_api_key
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()

# Set up API keys
os.environ["OPENAI_API_KEY"] = openai_api_key
print(os.environ["OPENAI_API_KEY"])

llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
# Initialize the prompt template
template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "You are a AWS helper. Based on user questions, create a reply to that."
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = PromptTemplate(template)

# Rebuild the storage context and load the index
storage_context = StorageContext.from_defaults(persist_dir="./aws_overview_01")
index = load_index_from_storage(storage_context, index_id="aws01")
query_request = index.as_query_engine(text_qa_template=qa_template)

# Streamlit interface
st.title("AWS Helper")

# Input for user question
query_str = st.text_input("Enter your question about AWS:")

# Button to submit the question
if st.button("Get Answer"):
    if query_str:
        response = query_request.query(query_str)
        
        # Display the response with structured formatting
        st.markdown("**Answer:**")
        
        with st.expander("Click to view the full answer", expanded=True):
            st.write(response)  # Basic display, can also be `st.markdown(response)` if markdown is needed

            # Optionally, display in a code block or text area for better readability
            st.code(response, language="markdown")  # Use language="text" if markdown rendering isn't needed

    else:
        st.warning("Please enter a question.")
