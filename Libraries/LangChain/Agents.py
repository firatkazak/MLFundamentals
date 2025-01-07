from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

loader = WebBaseLoader("https://blog.langchain.dev/langgraph/")
docs = loader.load()
print("Döküman Çıktısı 1: ", docs)

embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
print("Döküman Çıktısı 2:", documents)

vector = Chroma.from_documents(documents, embeddings)
retriever = vector.as_retriever()

question1 = retriever.get_relevant_documents("what is LangGraph?")[0]
print("Sorunun Cevabı: ", question1)

retriever_tool = create_retriever_tool(retriever=retriever,
                                       name="langgraph_search",
                                       description="Search for information about LangGraph. For any question\
                                    about LangGraph, you must use this tool!",
                                       )

search = TavilySearchResults()
question2 = search.invoke("what is the weather in istanbul?")
print("Sorunun Cevabı: ", question2)

tools = [retriever_tool, search]

model = ChatOpenAI(temperature=0)

prompt = hub.pull("hwchase17/openai-functions-agent")
print("Prompt Mesajı: ", prompt.messages)

agent = create_openai_functions_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               verbose=True
                               )

response1 = agent_executor.invoke({"input": "Hi!"})
print("Dönüş 1: ", response1)
response2 = agent_executor.invoke({"input": "what is the weather in london?"})
print("Dönüş 2: ", response2)

chat_history = [
    HumanMessage(content="Can I use LangGraph for agent runtimes?"),
    AIMessage(content="Yes!"),
]

response3 = agent_executor.invoke(
    {
        "chat_history": chat_history,
        "input": "Tell me how",
    }
)
print("Dönüş 3: ", response3)
