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
print("API Key Loaded:", os.environ["OPENAI_API_KEY"])

# Initialize the OpenAI LLM
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
st.set_page_config(page_title="AWS Helper", page_icon="🤖", layout="wide")
st.title("🤖 AWS Helper")
st.subheader("Get instant insights from AWS documentation")

# Input for user question
query_str = st.text_input("💡 Enter your question about AWS:", placeholder="e.g., What is the latest with AWS?")

# Button to submit the question
if st.button("Get Answer"):
    if query_str:
        with st.spinner("Fetching the answer..."):
            response = query_request.query(query_str)

        # Enhanced display of the response
        st.success("Here is your answer:")
        st.markdown("### **Answer:**")
        st.markdown(f"📝 {response}", unsafe_allow_html=True)

        # Additional UI options for better user experience
        with st.expander("🔍 View Full Answer"):
            st.text_area("Full Answer", value=str(response), height=200)

        # Download button for the response
        st.download_button(
            label="📥 Download Answer",
            data=str(response),
            file_name="aws_answer.txt",
            mime="text/plain",
        )

    else:
        st.warning("❗ Please enter a question.")
