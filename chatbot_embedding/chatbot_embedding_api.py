import openai
import pandas as pd
import numpy as np

# OpenAI 출처 https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

openai.api_key = "본인의 api키를 입력"

datafile_path = "C:/소스 위치/chatbot_embedding/chatbot_source.csv"

df = pd.read_csv(datafile_path)
#df.head()
df = df.set_index(["title", "heading"])
#print(f"{len(df)} rows in the data.")

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo-0613"

########################### CSV데이터를 읽어와 임베딩하여 dict데이터 준비 ###########################

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    """
    CSV데이터의 content를 OpenAI에 api로 보내서 임베딩 데이터를 받아와 float데이터들만 반환
    """
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    CSV에서 한줄씩 content를 api 임베딩 처리하는 함수로 전달 반환할때는 위에서 index 설정한 값과 함께 임베딩 float데이터를 dict로 반환
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

document_embeddings = compute_doc_embeddings(df)

# 준비된 dict 데이터의 첫번째 값을 출력해보기
#example_entry = list(document_embeddings.items())[0]
#print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

##################################### CSV데이터 준비 완료 #####################################


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


############################# 정렬된 데이터 5개만 구분자를 추가해서 반환 #############################

SEPARATOR = "\n* "

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    질문과 준비된 dict를 비교하여 정렬
    """
    most_relevant_document_sections = order_by_similarity(question, context_embeddings)
    
    """
    정렬된 값의 구분자를 각각의 content 앞에 추가해서 list로 반환
    """
    chosen_sections = []
    chosen_sections_indexes = []
        
    for idx in range(0, 5):
        document_section = df.loc[most_relevant_document_sections[idx][1]]

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(most_relevant_document_sections[idx][1]))
    
    # 반환되는 값 정보
#    print(f"Selected {len(chosen_sections)} document sections:")
#    print("\n".join(chosen_sections_indexes))
    
    return chosen_sections

######################################### 추가 완료 #########################################


############# 질문에 대해 위의 작업을 진행하고 정렬된 값과 질문을 ChatGPT에 같이 보내 답을 반환 #############

def answer_with_gpt_3(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array]
) -> str:
    """
    시스템의 역할을 부여
    """
    messages = [
        {"role" : "system", "content":"You are a PRIVATE chatbot, only answer the question by using the provided context. If your are unable to answer the question using the provided context, say '죄송합니다. 대답할 수 없는 질문입니다.'"}
    ]
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    """
    위의 정렬되고 추가된 값을 하나의 str로 만들고 구분자를 추가한 뒤 질문을 붙여줌
    """
    context= ""
    for article in prompt:
        context = context + article 

    context = context + '\n\n --- \n\n + ' + query

    messages.append({"role" : "user", "content":context})
    """
    temperature를 0.0으로 해야 주어진 데이터를 기반으로 답변을 함
    """
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=2000
        )

    return '\n' + response['choices'][0]['message']['content']

######################################### 답변 완료 #########################################

prompt = "오늘 매출은?"
response = answer_with_gpt_3(prompt, df, document_embeddings)
print(prompt + response)
