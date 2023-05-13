import openai
import pandas as pd

# COMPLETIONS_MODEL = "gpt-4"
# https://openai.com/waitlist/gpt-4-api  <- 여기서 GPT-4 사용 등록 요청 후 승인되면 사용가능, 그 전까지 3.5 사용
MODEL = "gpt-3.5-turbo"

COMPLETIONS_API_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 4097,
    "model": MODEL,
}

df = pd.read_csv('C:/소스 위치/lotto_combined.csv')


def get_lottoNumbers(question: str, df: pd.DataFrame) -> str:
    
    chosen_sections = []
    
    # 토큰이 4097을 넘어서 90여개정도만 보낼 수 있어서 최근꺼로 보냄
    # 토큰 초과 에러 메시지
    # This model's maximum context length is 4097 tokens. However, your messages resulted in 37333 tokens. Please reduce the length of the messages
    for i in range(960, len(df.index)):
        document_section = df.loc[i]

        chosen_sections.append(document_section.combined.replace("\n", " "))
        
    return chosen_sections


def answer_with_gpt(
    query: str,
    df: pd.DataFrame
) -> str:
    messages = [
        {"role" : "system", "content":"You are a lottery prediction chatbot, only answer the question by using the provided context."}
    ]
    lottoArray = get_lottoNumbers(
        query,
        df
    )

    context= ""
    for lottoNum in lottoArray:
        context = context + lottoNum 

    context = context + '\n\n --- \n\n + ' + query
    # print(context)

    messages.append({"role" : "user", "content":context})
    # print(messages)
    
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages
        )

    return '\n' + response['choices'][0]['message']['content']


response = answer_with_gpt('Predict lottery6', df)
print(response)
