import os
# from dotenv import load_dotenv, find_dotenv
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.llms import AzureOpenAI
# from langchain.document_loaders import DirectoryLoader,PyPDFLoader
# from langchain.document_loaders import UnstructuredExcelLoader
# from langchain.vectorstores import DocArrayInMemorySearch
# from langchain.memory import ConversationBufferMemory
# from IPython.display import display, Markdown
# import pandas as pd
# import gradio as gr
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain import PromptTemplate
# from langchain.vectorstores import Chroma
from langchain.agents.tools import Tool
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain import OpenAI, VectorDBQA
# from langchain.chains.router import MultiRetrievalQAChain
import streamlit as st
from streamlit_chat import message
# from langchain.document_loaders import UnstructuredPDFLoader
# _ = load_dotenv(find_dotenv())

os.environ["GOOGLE_API_KEY"] = "AIzaSyD6lORTrf5wLPP6wR6keH6yhP2Kwd-A1r4"
llm = GooglePalm(model_name = "models/text-bison-001",temperature = 0.1)
embeddings = GooglePalmEmbeddings(model_name="models/embedding-gecko-001")

bcar_retriever = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='Basel Capital Adequacy Reporting (BCAR) 2023_index')
# .as_retriever()
bmo_retriver = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='bmo_ar2022_index')
# .as_retriever()
creditirb_retriever = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='Capital Adequacy Requirements (CAR) Chapter 5 Credit Risk Internal Ratings Based Approach_index')
# .as_retriever()
creditstd_retriever = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='Capital Adequacy Requirements (CAR) Chapter 4  Credit Risk Standardized Approach_index')
# .as_retriever()
nbc_retriever = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='NATIONAL BANK OF CANADA_ 2022 Annual Report_index')
# .as_retriever()
smsb_retriever = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name='SMSB_index')
# .as_retriever()

indices = [bcar_retriever,bmo_retriver,creditirb_retriever,creditstd_retriever,nbc_retriever,smsb_retriever]

for index in indices[1:]:
    indices[0].merge(index)

# qa_bcar = RetrievalQA.from_chain_type(llm=llm, retriever=bcar_retriever, verbose=True)
# qa_bmo = RetrievalQA.from_chain_type(llm=llm, retriever=bmo_retriver, verbose=True)
# qa_creditirb = RetrievalQA.from_chain_type(llm=llm, retriever=creditirb_retriever, verbose=True)
# qa_creditstd = RetrievalQA.from_chain_type(llm=llm, retriever=creditstd_retriever, verbose=True)
# qa_smsb = RetrievalQA.from_chain_type(llm=llm, retriever=smsb_retriever, verbose=True)
# qa_nbc = RetrievalQA.from_chain_type(llm=llm, retriever=nbc_retriever, verbose=True)

# tools = [
#     Tool(
#         name = "BCAR",
#         func=qa_bcar.run,
#         description="useful for when you need to find answer regarding bcar different categories and schedules"
#     ),
#     Tool(
#         name="BMO Annual Report",
#         func=qa_bmo.run,
#         description="useful for when you need to find details about BMO bank like category it follows, fiscal year end etc"
#     ),
#     Tool(
#         name="Credit Risk –Internal Ratings Based Approach",
#         func=qa_creditirb.run,
#         description="useful for when you need to find details about Credit Risk –Internal Ratings Based Approach "
#     ),
#     Tool(
#         name="Credit Risk –Standardized Approach",
#         func=qa_creditstd.run,
#         description="useful for when you need to find details about Credit Risk –Standardized Approach "
#     ),
#     Tool(
#         name="SMSB",
#         func=qa_smsb.run,
#         description="useful for when you need to find details about SMSB that is one category approach among BCAR"
#     ),
#     Tool(
#         name="National Bnak Of Canada Annual Report",
#         func=qa_nbc.run,
#         description="useful for when you need to find details about National Bank of Canada like category it follows, fiscal year end etc"
#     ),
# ]
# planner = load_chat_planner(llm)
# executor = load_agent_executor(llm, tools, verbose=True)
# agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent = RetrievalQA.from_chain_type(llm=llm, retriever=bcar_retriever.as_retriever(), verbose=True)

# generated stores AI generated responses
st.title("BMO Chatbot")
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = []
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
user_input = get_text()
if user_input:
    output = agent.run(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
if 'generated' in st.session_state:
    for i in range(len(st.session_state['generated'])-1,-1,-1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))


# agent.run("Which reports bank BMO has to send to OSFI for BCAR Credit Risk?")