from langgraph.graph import START, END, StateGraph
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenRouter(
    model="gpt-4.1-mini"
)

class PostState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    revision_count: int

def writer_node(state: PostState) -> PostState:
    prompt = f"You are a viral LinkedIn content creator. Write a post based on the topic provided. Use a strong hook, plenty of white spaces, and an engaging CTA. If you see a critique, rewrite the post to address the specific feedback."
    count = state.get('revision_count', 0)
    messages = [HumanMessage(content=prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {'messages': [AIMessage(content=response.content)], 'revision_count': count+1}

def critique_node(state: PostState) -> PostState:
    last_post = state['messages'][-1].content

    prompt=f"""
    Review the LinkedIn Post based on below points:
    1. Is there a 'hook' in the first 2 lines?
    2. Is it easy to read on mobile (short sentences)?
    3. Is there a question at the end to drive comments?
    4. Is it creating curiousity among readers?
    5. Is there a slight humour in the post?

    If the post is perfect, respond ONLY with 'READY'.
    Otherwise, provide a bulleted list of what to fix.
    
    Post: {last_post}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {'messages': AIMessage(content=response.content)}

def route_post(state: PostState) -> Literal["writer", "end"]:
    last_feedback = state['messages'][-1].content

    if "READY" in last_feedback.upper() or state['revision_count']>=3:
        return "end"
    return "writer"

builder = StateGraph(PostState)
builder.add_node("writer", writer_node)
builder.add_node("critique", critique_node)

builder.add_edge(START, "writer")
builder.add_edge("writer", "critique")
builder.add_conditional_edges("critique", route_post, {"writer": "writer", "end": END})

graph = builder.compile(checkpointer=InMemorySaver())

image = graph.get_graph().draw_mermaid_png()
with open("iterative.png", mode="wb") as f:
    f.write(image)

topic = {"messages": [HumanMessage(content="Write a post about why AI agents are the future of software development.")]}

for message_chunk, metadata in graph.stream(topic, stream_mode="messages", config={"configurable": {"thread_id": "1"}}):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)