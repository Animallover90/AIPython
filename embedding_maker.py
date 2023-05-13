import openai
import pandas as pd

datafile_path = "C:/소스 위치/lotto_combined_source.csv"

## 아래 주석은 맨 밑의 번호를 각각 embedding 처리 하가 위한 path
# datafile_path = "C:/소스 위치/lotto_source.csv"
# datafile_path = "C:/소스 위치/every_embedded_lotto2.csv"
# datafile_path = "C:/소스 위치/every_embedded_lotto3.csv"
# datafile_path = "C:/소스 위치/every_embedded_lotto4.csv"  
# datafile_path = "C:/소스 위치/every_embedded_lotto5.csv"

df = pd.read_csv(datafile_path)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

## combined 폴더용 임베딩 처리
df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(str(x), model='text-embedding-ada-002'))
df.to_csv('C:/저장 위치/embedded_lotto.csv', index=False)


## each_embedding 폴더용 임베딩 처리
## 이 아래는 순서대로 만들어진 파일에 추가하면서 Num1~ 6까지 순서대로 embedding 처리(각각의 값을 이용하여 분석하기 위해)
## 하나씩 주석을 풀면서 하는 이유는 한번에 했다가 엄청 오래 걸려서 결국 에러 떨어지는 경우 발생하기 때문

# df['num1_embedding'] = df.Num1.apply(lambda x: get_embedding(str(x), model='text-embedding-ada-002'))
# df.to_csv('C:/저장 위치/lotto_source.csv', index=False)

# df['num2_embedding'] = df.Num2.apply(lambda x: get_embedding(str(x), model='text-embedding-ada-002'))
# df.to_csv('C:/저장 위치/every_embedded_lotto2.csv', index=False)

# df['num3_embedding'] = df.Num3.apply(lambda x: get_embedding(str(x), model='text-embedding-ada-002'))
# df.to_csv('C:/저장 위치/every_embedded_lotto3.csv', index=False)

# df['num4_embedding'] = df.Num4.apply(lambda x: get_embedding(str(x), model='text-embedding-ada-002'))
# df.to_csv('C:/저장 위치/every_embedded_lotto4.csv', index=False)

# df['num5_embedding'] = df.Num5.apply(lambda x: get_embedding(str(x), model='text-embedding-ada-002'))
# df.to_csv('C:/저장 위치/every_embedded_lotto5.csv', index=False)

# df['num6_embedding'] = df.Num6.apply(lambda x: get_embedding(str(x), model='text-embedding-ada-002'))
# df.to_csv('C:/저장 위치/every_embedded_lotto6.csv', index=False)