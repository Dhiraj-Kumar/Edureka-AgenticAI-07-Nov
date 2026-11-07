from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model="gpt-4.1-mini")

response = model.invoke("What is the capital of India")

print(response.content)