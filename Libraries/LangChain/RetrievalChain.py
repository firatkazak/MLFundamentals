from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

load_dotenv()

model = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}"),
    ]
)

output_parser = StrOutputParser()
chain = prompt | model | output_parser
response1 = chain.invoke({"input": "What is LangGraph?"})
print("Yanıt 1: ", response1)

embeddings = OpenAIEmbeddings()
loader = WebBaseLoader("https://blog.langchain.dev/langgraph/")
docs = loader.load()
print("Bloğun İçindekiler 1: ", docs)

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
print("Bloğun İçindekiler 2: ", documents)

vector = Chroma.from_documents(documents, embeddings)
retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke(
    {
        "input": "What is LangGraph?"
    }
)
print("Response 1: ", response)

print("Cevap 1: ", response["answer"])

my_doc = Document(page_content="LangGraph is module built on top of LangChain to better enable creation of cyclical graphs, often needed for agent runtimes.")

response2 = document_chain.invoke(
    {
        "input": "What is LangGraph?",
        "context": [my_doc],
    }
)
print("Son Yanıt: ", response2)
