from langchain_fireworks.chat_models import ChatFireworks
from dotenv import load_dotenv


load_dotenv()
model = ChatFireworks(model_name="accounts/fireworks/models/gpt-oss-120b")

result = model.invoke("hello")
print(result.content)