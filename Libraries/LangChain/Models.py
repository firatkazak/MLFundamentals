from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAI

load_dotenv()

llm1 = ChatOpenAI(model_name="gpt-3.5-turbo",
                  temperature=0.6,
                  )

text = "What is the capital of Turkey"

response = llm1.invoke(text)
print("OpenAI Response: ", response)

hf = HuggingFaceEndpoint(
    model="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7,
    do_sample=False,
    repetition_penalty=1.03,
)

output = hf.invoke(text)
print("HF Çıktısı: ", output)

text = "What is the capital of USA"
output = hf.invoke(text)
print("ABD'nin Başkenti: ", output)

text = "What is the capital of Russia"
output = hf.invoke(text)
print("Rusya'nın Başkenti: ", output)

text = "What is the capital of Turkey"
output = hf.invoke(text)
print("Türkiye'nin Başkenti: ", output)

output = hf.invoke("Can you tell me what deep learning is?")
print("Deep Learning Sorusunun Cevabı: ", output)

text = "Tell me a joke about Fenerbahçe."
messages = [HumanMessage(content=text)]
print("Şaka: ", messages)

llm2 = OpenAI()
chat_model = ChatOpenAI()
print(llm2.invoke(text))

print(chat_model.invoke(messages))
