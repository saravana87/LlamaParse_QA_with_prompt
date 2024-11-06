import nest_asyncio
import yaml
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
import os
nest_asyncio.apply()

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# API access
llama_cloud_API_KEY = config["llama_cloud_api_key"]
os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

# Use dynamic model selection from config
embed_model = OpenAIEmbedding(model=config["openai_embedding_model"])
llm = OpenAI(model=config["openai_llm_model"])

Settings.llm = llm
Settings.embed_model = embed_model
__all__ = ["embed_model", "llm", "Settings","llama_cloud_API_KEY"]