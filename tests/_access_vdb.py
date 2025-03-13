from app.core.langchain_module.rag import VectorStore

vdb = VectorStore(collection_name="state_map")
print(vdb.get_collection_info())
