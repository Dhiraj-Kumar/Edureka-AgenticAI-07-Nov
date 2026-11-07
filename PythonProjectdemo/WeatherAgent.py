from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from dotenv import load_dotenv
import requests
import os

load_dotenv()

model = ChatOpenRouter(model="gpt-4.1-mini")

@tool
def getWeatherData(city: str):
    """
    Get the current weather for a city using openweathermap api
    """
    url="https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": os.getenv("WEATHER_API_KEY"),
        "units": "metric"
    }

    response = requests.get(url, params)

    if response.status_code != 200:
        return f"Unable to fetch data from weather api"
    return response.json()

with SqliteSaver.from_conn_string("checkpointdb.sqlite") as checkkpointer:
    checkkpointer.setup()
    agent = create_agent(
        model=model,
        system_prompt="You are an experienced AI weather assistant who answers queries about latest weather information by calling external tools. You give your responses with a bit of humour. Use emojis to make your responses looks good.",
        checkpointer=checkkpointer,
        tools=[getWeatherData],
        middleware=[
            SummarizationMiddleware(
                # trigger=("tokens", 50000)
                trigger=("fraction", 0.6),
                keep=("messages", 10)
            )
        ]
    )

    while True:
        user_input = input("You: ")
        if user_input.lower()=="exit":
            break
        response = agent.invoke(
            {
                "messages": {
                    "role": "user",
                    "content": user_input
                }
            }, config={"configurable": {"thread_id": "1"}}
        )
        print(f"AI: {response['messages'][-1].content}")