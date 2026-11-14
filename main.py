import os
from dotenv import load_dotenv
from langchain_fireworks.chat_models import ChatFireworks
from langchain_fireworks.embeddings import FireworksEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from DocumentLoaders import load_pdf_file, load_txt_file, load_web_page
from TextSplitters import split_as_character, split_as_tokens, split_as_recursive_character
load_dotenv()

embedding_model = FireworksEmbeddings(
    model="accounts/fireworks/models/qwen3-embedding-8b",
    fireworks_api_key=os.getenv("FIREWORKS_API_KEY"),
)

vectorstore = Chroma(
    embedding_function = embedding_model,
    persist_directory="chroma_db",
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs={"k": 4, "lambda_mult": 0.25, "fetch_k": 10}
)

template = ChatPromptTemplate.from_messages([
    ("system", "You are an hashtag creator, create 5 hashtags only no other texts"),
    ("user", "{content}")
    ])

model = ChatFireworks(
    model_name="accounts/fireworks/models/gpt-oss-120b",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

messages = template.format_messages()
result = model.invoke(messages)
print(result.content)


