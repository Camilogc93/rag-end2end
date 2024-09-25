import os

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import chainlit as cl
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser
from typing import cast
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import qdrant_client

load_dotenv()


quadran=os.environ["quadran_key"]
endpoint=os.environ["endpoint"]

# Define a different ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
            """
            You are an AI assistant that provides helpful answers to user queries.
            """),
        ("user", "{question}\n"),
    ]
)


@cl.on_chat_start
def main():
    model_name = "CamiloGC93/bge-large-en-v1.5-etical"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)   
    embeddings = hf
   

    client = qdrant_client.QdrantClient(
    endpoint,
    api_key=quadran,
     https=True, 
     port=6333
)
    qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="EticalAI",
    url=endpoint,
    api_key=quadran
)




    retriever = qdrant.as_retriever()

    RAG_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Context: {context}
Question: {question}
"""

    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")
    rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | rag_prompt | llm | StrOutputParser()
)
    print("data on database")
    cl.user_session.set("rag_chain", rag_chain)

 



@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("rag_chain"))  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
