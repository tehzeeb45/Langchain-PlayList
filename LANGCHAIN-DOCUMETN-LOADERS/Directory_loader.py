from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# Lazy load (generator)
docs = loader.lazy_load()

# Loop se iterate karna hoga
for document in docs:
    print(document.metadata)      # sirf metadata
    # print(document.page_content)  # agar content bhi chahiye
