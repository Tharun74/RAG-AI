import os
from dotenv import load_dotenv
from langchain_fireworks.chat_models import ChatFireworks
from langchain_core.prompts import ChatPromptTemplate
from document_loaders import load_pdf_file, load_txt_file, load_web_page
from text_spliters import split_as_character, split_as_tokens, split_as_recursive_character
load_dotenv()

template = ChatPromptTemplate.from_messages([
    ("system", "You are an hashtag creator, create 5 hashtags only no other texts"),
    ("user", "{content}")
    ])

model = ChatFireworks(
    model_name="accounts/fireworks/models/gpt-oss-120b",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

docs = load_txt_file("resources/article1.txt")
chunks = split_as_recursive_character(docs)

for i in chunks:
    print(i.page_content)
    print()
    print()

# messages = template.format_messages(content = docs[0].page_content)
# result = model.invoke(messages)
# print(result.content)