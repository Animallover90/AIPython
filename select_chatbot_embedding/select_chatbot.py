import openai
import pandas as pd
import numpy as np
import ast
import requests
from datetime import datetime

openai.api_key = "api키 입력"

############### 임베딩 처리된 데이터 준비 및 임베딩 데이터를 float로 변환 시킨 데이터도 준비 ###############

datafile_path = "./embedded_data.csv"

df = pd.read_csv(datafile_path)
#print(df.head())
df = df.set_index(["title", "heading"])
#print(f"{len(embedded_data)} rows in the data.")

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo-16k"

document_embeddings = dict()
for idx, r in df.iterrows():
    document_embeddings[idx] = ast.literal_eval(r.get('embedding'))
    
# 준비된 dict 데이터의 첫번째 값을 출력해보기
#example_entry = list(document_embeddings.items())[0]
#print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

###################################### 데이터 준비 완료 ######################################


#################################### 질문 str을 임베딩처리 ####################################

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    """
    유저의 질문을 OpenAI에 api로 보내서 임베딩 데이터를 받아와 float데이터들만 반환
    """
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

######################################## 임베딩 완료 ########################################


################### 질문을 임베딩하고 준비된 임베딩 데이터와 비교하여 비슷한 순서대로 정렬 ###################

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    임베딩 된 질문과 준비된 dict의 float를 numpy의 dot를 사용해서 각 자리수를 곱해서 합을 반환
    """
    return np.dot(np.array(x), np.array(y))

def order_by_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    질문을 임베딩처리, 위에서 미리 준비한 dict 데이터에서 float를 한줄씩 뽑아 질문과 값을 비교 후 값이 높은것부터 정렬
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

# 월간 매출은? 이라는 질문을 임베딩하여 비슷한 데이터를 위에서부터 5개 출력해보기
#print(order_by_similarity("월간 매출은?", document_embeddings)[:5])

######################################### 정렬 완료 #########################################


################################### api 통신하여 데이터 조회 ###################################

def post_request(data: dict, url: str) -> dict:
    """
    유저의 질문을 OpenAI에 api로 보내서 임베딩 데이터를 받아와 float데이터들만 반환
    """
    headers = {'Content-type': 'application/json'}
    try:
        response = requests.post("http://127.0.0.1:8080/"+url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json().get('result')
        else:
            print('데이터 조회에 실패하였습니다.')
            return None
    except:
        print('데이터 조회 서버 연결에 실패하였습니다.')
        return None

######################################## 임베딩 완료 ########################################


################# 정렬된 데이터 5개 중에서 선택된 내용의 url을 이용해 데이터 조회하여 반환 #################

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> dict:
    """
    질문 임베딩 값과 준비된 dict를 비교하여 정렬
    """
    most_relevant_document_sections = order_by_similarity(question, context_embeddings)
    ## >> [(0.9045000063975488, ('유저의 아이템 구매 기록', 2)), (0.8908295902792714, ('유저의 현금 아이템 구매 기록', 4)), (0.8719414680464882, ('유저의 현금 아이템 환불 기록', 5)), (0.8473006060437999, ('유저의 현금 충전 기록', 3)), (0.8419848894806348, ('유저의 로그인 기록', 1)), (0.8364480399668113, ('최근 1주일 매출', 9)), (0.8335642299306989, ('저번달 1달간 매출', 10)), (0.8256427842497658, ('저번달 접속자 수', 7)), (0.8140220223067189, ('작년 월 평균 접속자 수', 8)), (0.8132117026958139, ('어제 접속자 수', 6))]
    """
    정렬된 값의 상위 5개를 보여주고 번호를 입력 받아 해당 번호의 url을 가져와 통신(param 데이터도 입력 받고 data를 전달)
    """
    print("아래 리스트 중 해당하는 내용을 선택하세요.")
    for idx in range(0, 5):
        print(f"{idx+1}. {most_relevant_document_sections[idx][1][0]}")
    print("6. 취소")
    
    cont = True
    req = 0
    result_data = dict()
    question = ""
    while cont:
        try:
            req = int(input('번호 입력 : '))
            cont = False
        except:
            print('번호를 입력하시기 바랍니다.')
    
    if req == 6:
        return req
        
    if 1<=req<=5:
        document_section = df.loc[most_relevant_document_sections[req-1][1]]
        question = document_section.question
        if document_section.param != 'none':
            params = document_section.param.split(';')
            data = dict()
            for idx in params:
                req = str(input(f'{idx} 정보 입력 : '))
                data[idx] = req
            result_data = post_request(data, document_section.get('url'))
        else:
            result_data = post_request(None, document_section.get('url'))
    else:
        print('해당하는 번호가 없습니다. 다시 확인하시기 바랍니다.')
        return None

    if result_data is None:
        return result_data
    if result_data == 'No Data':
        return result_data
    
    result_data.append({'today' : str(datetime.now().today().strftime("%Y-%m-%d"))})
    result_data.append({'question' : question})
    return result_data

######################################### 추가 완료 #########################################


############# 질문에 대해 위의 작업을 진행하고 정렬된 값과 질문을 ChatGPT에 같이 보내 답을 반환 #############

def answer_with_gpt(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array]
) -> str:
    """
    시스템의 역할을 부여
    """
    messages = [
        {"role" : "system", "content":"You are a PRIVATE chatbot, only answer the question by using the provided context. If calculations are required, include the results of calculations in the answer. Do not answer about code. Do not answer methods. Always return Korean"}
    ]
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    if prompt is None:
        return prompt
    if prompt == 'No Data':
        return prompt
    if prompt == 6:
        return prompt
    
    messages.append({"role" : "user", "content":str(prompt)})
    """
    temperature를 0.0으로 해야 주어진 데이터를 기반으로 답변을 함
    """
    print(f'GPT에 보내는 내용 : {messages}')
    try:
        response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=6000
        )
    except:
        print('GPT 통신에 실패하였습니다.')
        return 6

    return '\n' + response['choices'][0]['message']['content']

########################################## 답변 완료 #########################################


########################################## 질문 시작 #########################################

while True:
    req = str(input('질문을 입력하세요.\n끝내기 위해서는 exit를 입력하세요.\n입력 : '))
    if req == 'exit':
        breake
    else:
        response = answer_with_gpt(req, df, document_embeddings)
        if response is None:
            print('\n질문에 대해 답을 할 수 없습니다.\n\n')
        elif response == 6:
            print('\n\n')
        elif response == 'No Data':
            print('\n데이터가 없습니다.\n\n')
        else:
            print('\n' + response + '\n\n')

########################################## 질문 완료 #########################################
