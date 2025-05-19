from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

pdf_path = 'PPC_TSI_EaD.pdf'
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
pages = pages[:3]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
)

texts = text_splitter.split_documents(pages)

db = FAISS.from_documents(texts, OllamaEmbeddings(model="nomic-embed-text"))

model = OllamaLLM(model="llama3")
retriever = db.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")

query = "Qual o nome do curso?"
result = qa_chain.invoke(query)
print(result)