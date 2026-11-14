import os
from dotenv import load_dotenv
from DocumentLoaders import load_txt_file
from TextSplitters import split_as_recursive_character
from langchain_community.vectorstores import Chroma
from langchain_fireworks import FireworksEmbeddings


load_dotenv()

docs = []
docs.extend(load_txt_file("resources/article1.txt"))
docs.extend(load_txt_file("resources/article2.txt"))
docs.extend(load_txt_file("resources/article3.txt"))

chunks = split_as_recursive_character(docs)

embeddings = FireworksEmbeddings(
    model="accounts/fireworks/models/qwen3-embedding-8b",
    fireworks_api_key=os.getenv("FIREWORKS_API_KEY"),
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="articles",
    persist_directory="chroma_db",
)

result = vectorstore.similarity_search("What did trump say about nvidia", k = 2)

for r in result:
    print(r)
    



