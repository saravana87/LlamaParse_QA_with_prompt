from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from llamacloud_init import llama_cloud_API_KEY
from llama_index.core import VectorStoreIndex

parser = LlamaParse(    
    result_type="text",  # "markdown" and "text" are available
    verbose=True,
    api_key = llama_cloud_API_KEY
)

file_extractor = {".pdf": parser,".txt": parser}
documents = SimpleDirectoryReader(
    "./Sfiles", file_extractor=file_extractor
).load_data()
index = VectorStoreIndex.from_documents(documents)
index.set_index_id("aws01")
index.storage_context.persist("./aws_overview_01")