from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)

prompt_template.format(
    adjective="funny",
    content="Elon Musk"
)

llm = ChatOpenAI()

response = llm.invoke("Give me a suggestion for the main course\
 for today's lunch.")

print("Yanıt 1: ", response.content)
# Yanıt 1:  How about a delicious grilled chicken Caesar salad with homemade dressing and crunchy croutons? It's a light and refreshing option that is sure to satisfy your hunger.


prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}."
)

model = ChatOpenAI()

output_parser = StrOutputParser()
chain = prompt | model | output_parser
response_context1 = chain.invoke({"topic": "Elon Musk"})
print("Yanıt 2: ", response_context1)
# Yanıt 2:  Why did Elon Musk break up with his car?
# Because it couldn't handle his commitment issues!

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are an English-Turkish translator that\
 return whatever the user says in Turkish."),
     ("user", "{input}")]
)

chain = prompt | model | output_parser

response_context2 = chain.invoke({"input": "To be or not to be!"})
print("Yanıt 3: ", response_context2)
# Yanıt 3:  Olmak ya da olmamak, işte bütün mesele bu!
