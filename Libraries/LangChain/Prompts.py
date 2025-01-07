from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate

load_dotenv()

prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
)

prompt.format(topic="Bill Gates")

llm = ChatOpenAI()
chain = prompt | llm

result1 = chain.invoke({"topic": "Bill Gates"})
print("Şaka 1: ", result1)
# Şaka 1:  content='Why did Bill Gates get a job at the bakery? Because he kneaded the dough!' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 14, 'total_tokens': 33, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-a545e665-abf9-4f6e-8764-690463265f65-0' usage_metadata={'input_tokens': 14, 'output_tokens': 19, 'total_tokens': 33, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

output_parser = StrOutputParser()
chain = prompt | llm | output_parser
result2 = chain.invoke({"topic": "Bill Gates"})
print("Şaka 2: ", result2)
# Şaka 2:  Why did Bill Gates get rid of his old computer?
# Because it couldn't handle his Windows updates!

result3 = chain.invoke({"topic": "Taylor Swift"})
print("Şaka 3: ", result3)
# Şaka 3:  Why did Taylor Swift go to the doctor? Because she had too many "Bad Blood" cells!

template = "You are a helpful assistant that traslates\
 {input_language} to {output_language}."

human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

result4 = chat_prompt.format_messages(input_language="English",
                                      output_language="Turkish",
                                      text="I love programming."
                                      )
print("4. Çıktı: ", result4)
# 4. Çıktı:  [SystemMessage(content='You are a helpful assistant that traslates English to Turkish.', additional_kwargs={}, response_metadata={}), HumanMessage(content='I love programming.', additional_kwargs={}, response_metadata={})]

chain = chat_prompt | llm | output_parser

result5 = chain.invoke({
    "input_language": "English",
    "output_language": "Turkish",
    "text": "I love programming."
})
print("5. Çıktı: ", result5)
# 5. Çıktı:  Ben programlamayı seviyorum.
