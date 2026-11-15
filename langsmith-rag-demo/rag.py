from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1. Loading PDF document
loader = PyPDFLoader("./Documents/HR-Policy.pdf")
documents = loader.load()

# Step 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

# Step 3. Create instance for an Embedding model
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# Step 4. Store embeddings into Vector DB
PERSIST_DIR = "./chroma_db"
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorestore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    vectorestore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR
    )

# Step 5. Create a retriever
retriever = vectorestore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5}
)

# Step 6. Augmentation
prompt = PromptTemplate(
    template="""
    You are a helpful AI assistant.
    Answer the question using ONLY the context below
    
    Context:
    {context}
    
    Question:
    {question}
    """,
    input_variables=["context", "question"]
)

# Step 7. Initialize the LLM
llm =ChatOpenRouter(
    model="gpt-4o-mini"    
)

def rag_pipeline(inputs: dict):
# Step 8. Create a RAG pipeline (chain)
    question = inputs["question"]
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return {"answer": response.content, "context": context}


# Guard the demo call — otherwise it fires on every import from evaluate_rag.py
if __name__ == "__main__":
    result = rag_pipeline({"question": "How many leaves an employee can take in a year?"})
    print(result["answer"])