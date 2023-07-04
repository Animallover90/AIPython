import openai
import pandas as pd

openai.api_key = "api키 입력"

datafile_path = "./select_chatbot_source.csv"

df = pd.read_csv(datafile_path)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

## 임베딩 처리
df['embedding'] = df.content.apply(lambda x: get_embedding(str(x), model='text-embedding-ada-002'))
df.to_csv('./embedded_data.csv', index=False)
