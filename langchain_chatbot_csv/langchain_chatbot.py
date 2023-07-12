import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain, ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory
request_memory = ConversationBufferMemory(input_key='query', memory_key='chat_history')

http_url = "http://127.0.0.1:8080/"
os.environ['OPENAI_API_KEY'] = "api키 입력"
base_llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo-0613")

## 화면에 표시되는 타이틀과 텍스트필드
st.title('🦜🔗 Langchain chatbot (csv file)')
question = st.text_input('질문 내용을 작성해주세요.(날짜는 최대한 명확하게)') 

########### CSV 데이터 로드 및 embedding하여 vectorstore 저장 및 Chain 준비 ###########

loader = CSVLoader(file_path='./langchain_chatbot_source.csv')

data = loader.load()

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embeddings)

conversation_chain = ConversationalRetrievalChain.from_llm(
llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo-0613'),
retriever=vectorstore.as_retriever(), return_source_documents=True, verbose=True)

################################ CSV 데이터 준비 완료 ################################


######### vectorstore 저장소에서 유저 질문에 가까운 내용을 뽑아 각 CSV 컬럼의 값을 추출 #########

def conversational_chat(query)  -> list:
      
      chat_history = []
      result = conversation_chain({"question": query, "chat_history": chat_history})
      
      return result["source_documents"][0].page_content.split("\n")

def get_csv_extraction(question: str) -> dict:
      conversation_response = conversational_chat(question)
      with st.expander('CSV Extraction History'): 
        st.info(conversation_response)

      csv_schema = {
          "properties": {
              "title" : {"type": "string"},
              "heading" : {"type": "string"},
              "content" : {"type": "string"},
              "url" : {"type": "string"},
              "param" : {"type": "string"},
          }
      }

      csv_extraction_chain = create_extraction_chain(csv_schema, base_llm)

      csv_extraction_response = csv_extraction_chain.run(conversation_response)

      return csv_extraction_response[0]

############################### CSV 컬럼의 값 추출 완료 ###############################


##### 질문과 질문의 param을 추춘한 dict을 이용해 HTTP GET요청하여 데이터를 받은 후 GPT답변 반환 ####

def get_request_get_answer(question: str, url: str, params: str, is_contain_param = False) -> str:
           
      template = """ You are a PRIVATE chatbot, only answer the question by using the provided Between >>> and <<<.
      Extract the answer to the question '{query}' or say "not found" if the information is not contained.
      Always return in Korean.
      Use the format
      Extracted:<answer or "not data">
      >>> {requests_result} <<<
      Extracted:"""

      PROMPT = PromptTemplate(
          input_variables=["query", "requests_result"],
          template=template,
      )

      chain = LLMRequestsChain(llm_chain=LLMChain(llm=OpenAI(temperature=0.0, model_name="gpt-4"), prompt=PROMPT, memory=request_memory, verbose=True))
      if is_contain_param:
            inputs = {
               "query": question,
               "url": http_url + url + params,
            }
      else:
            inputs = {
               "query": question,
               "url": http_url + url,
            }

      response = chain(inputs)

      with st.expander('Request History'):
         st.info(response)

      return response["output"]

#################################### 답변 반환 함수 ####################################


################ 추출 값에서 param의 값이 있으면 유저의 질문에서 해당 값을 추출  ################
################ param의 key 값에 대한 내용이 질문에 없으면 정보 부족으로 반환  ################

def get_answer(question: str) -> str:
      csv_extraction_response = get_csv_extraction(question)
      if csv_extraction_response["param"] != "none":
            params = csv_extraction_response["param"].split(';')

            params_schema = dict()
            each_param = dict()

            for idx in params:
               each_param[idx] = {"type": "string"}

            params_schema["properties"] = each_param

            # params_llm=ChatOpenAI(temperature=0.0, model_name="gpt-4")
            params_chain = create_extraction_chain(params_schema, base_llm)

            params_response = params_chain.run(question)
            with st.expander('Params Extraction History'): 
               st.info(params_response)
               
            params_data = "?"
            for key, value in params_response[0].items():
                 params_data = params_data + key + "=" + value
                 if value == '':
                     st.warning(key + " 값에 대한 정보가 부족합니다. 필요한 정보와 함께 질문하시기 바랍니다.")
                     return "No data"
            
            ## HTTP GET 요청하여 데이터 받은 후 GPT에 질문 후 대답 반환
            answer = get_request_get_answer(question, csv_extraction_response["url"], params_data, True)
            return answer
      else:
            answer= get_request_get_answer(question, csv_extraction_response["url"], None)
            return answer

######################## 유저 질문에서 필요 param 데이터 추출 완료  ########################


####################################### 질문 시작 #####################################

# 화면의 텍스트 필드에 내용이 있을 경우 작동
if question: 
   answer = get_answer(question)

   if answer != "No data":
      with st.expander('Conversation History'): 
         st.info(request_memory.buffer)

   st.write(answer)

####################################### 질문 완료 #####################################
