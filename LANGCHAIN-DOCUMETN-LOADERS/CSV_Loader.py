from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="Data-Guide.csv", encoding="latin1")
data = loader.load()

print(len(data))
print(data[1])