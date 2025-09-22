from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('file-example_PDF_1MB.pdf')
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 30,
    chunk_overlap = 0,
    separator=''
)
result = splitter.split_documents(docs)
print(result[0].page_content)