from langgraph.graph import START, END, StateGraph
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenRouter(
    model="gpt-4.1-mini"
)

class ChatbotState(TypedDict):
    # messages: list[str]
    messages: Annotated[list[str], add_messages]

def ChatbotNode(state: ChatbotState) -> ChatbotState:
    result = llm.invoke(state['messages'])
    return {'messages': [result]}

graph = StateGraph(ChatbotState)
graph.add_node('ChatbotNode', ChatbotNode)
graph.add_edge(START, 'ChatbotNode')
graph.add_edge('ChatbotNode', END)

workflow = graph.compile(checkpointer=InMemorySaver())

image = workflow.get_graph().draw_mermaid_png()
with open("chatbot_graph.png", mode="wb") as f:
    f.write(image)

while True:
    user_input=input("You: ")
    if user_input.lower()=="exit":
        break
    response = workflow.invoke({'messages': user_input}, config={"configurable": {"thread_id": "1"}})
    print(response['messages'][-1].content)