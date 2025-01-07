from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
model = ChatOpenAI()
embeddings = OpenAIEmbeddings()
loader = WebBaseLoader("https://blog.langchain.dev/langgraph/")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
print("Döküman Çıktısı: ", documents)

vector = Chroma.from_documents(documents, embeddings)
retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search\
 query to look up in order to get information relevant to the conversaion")
    ]
)

retriver_chain = create_history_aware_retriever(model,
                                                retriever,
                                                prompt
                                                )

chat_history = [
    HumanMessage(content="Can I use LangGraph for agent runtimes?"),
    AIMessage(content="Yes!")
]

response1 = retriver_chain.invoke(
    {
        "chat_history": chat_history,
        "input": "Tell me how",
    }
)

print("Çıktı 1: ", response1)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user's questions based on the below\
 context: \n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ]
)

documents_chain = create_stuff_documents_chain(model, prompt)

retriver_chain = create_retrieval_chain(retriver_chain,
                                        documents_chain
                                        )

chat_history = [
    HumanMessage(content="Can I use LangGraph for agent runtimes?"),
    AIMessage(content="Yes!")
]

response2 = retriver_chain.invoke(
    {
        "chat_history": chat_history,
        "input": "Tell me how",
    }
)

print("Çıktı 2: ", response2)
print("Çıktı 2 CEVAP: ", response2["answer"])
