from langchain_community.document_loaders import PyPDFDirectoryLoader


DATA_PATH = "data"

loader = PyPDFDirectoryLoader(DATA_PATH)
docs = loader.load()
print(docs)