import os 
from langchain.llms import GooglePalm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from tqdm import tqdm

file_names = os.listdir("./data/")

os.environ["GOOGLE_API_KEY"] = "AIzaSyD6lORTrf5wLPP6wR6keH6yhP2Kwd-A1r4"
llm = GooglePalm(model_name = "models/text-bison-001",temperature = 0.1)
embeddings = GooglePalmEmbeddings(model_name="models/embedding-gecko-001")

# for file in tqdm(file_names):
#     loader = PyPDFLoader(f"./data/{file}")
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter()
#     texts = text_splitter.split_documents(documents)
#     docsearch = FAISS.from_documents(texts, embeddings)
#     docsearch.save_local(folder_path='FAISS_VS', index_name=f"{file.split('.')[0]}_index")
#     print(file.split(".")[0])

docsearch = FAISS.load_local(folder_path='./FAISS_VS', embeddings=embeddings, index_name=f"{file_names[0].split('.')[0]}_index")
retriever = docsearch.as_retriever()

print(retriever.get_relevant_documents("Which reports bank BMO has to send to OSFI for BCAR Credit Risk?"))