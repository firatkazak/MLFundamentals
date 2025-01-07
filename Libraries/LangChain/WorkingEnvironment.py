from dotenv import load_dotenv

load_dotenv()

# 1. Adım; Bu 3 paketi kurduk.
# pip install -qU langchain-openai => LangChain paketi.
# pip install -U langsmith openai => LangSmith paketi.
# pip install python-dotenv => API KEY'ini projeye direkt eklemek için paket.

# 2. adım.
# .env: Buraya OPENAI_API_KEY adında bir değişken tanımlayıp key'imizi buraya kaydettik.

# 3. adım.
# .gitignore: Key'imizi github'a yollamamak için gitignore dosyası oluşturduk.

# 4. adım
# Kodu çalıştırınca keyimiz projemize dahil edildi. Artık OpenAI'a, kendi key'imiz ile bağlandık.
