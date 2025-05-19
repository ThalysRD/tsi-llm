# Sistema de Recuperação de Informações com VectorDB e RetrievalQA

## Objetivo

Desenvolver um sistema capaz de responder perguntas sobre um documento PDF utilizando embeddings, banco de dados vetorial (FAISS) e o módulo RetrievalQA da biblioteca LangChain.

---

## Sumário

- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como funciona o sistema](#como-funciona-o-sistema)
- [Execução do código](#execução-do-código)
- [Personalização](#personalização)
- [Referências](#referências)

---

## Pré-requisitos

- Python 3.10 ou superior
- [Ollama](https://ollama.com/) instalado e rodando localmente
- Modelos baixados no Ollama:
  - `llama3` (para LLM)
  - `nomic-embed-text` (para embeddings)
- PDF de interesse (exemplo: `PPC_TSI_EaD.pdf`)

---

## Instalação

1. **Clone o repositório ou copie os arquivos para seu ambiente.**

2. **Instale as dependências:**

   ```bash
   pip install langchain langchain-community langchain-ollama faiss-cpu
   ```

3. **Baixe os modelos necessários no Ollama:**

   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

4. **Certifique-se de que o Ollama está rodando:**
   ```bash
   ollama serve
   ```

---

## Como funciona o sistema

O sistema segue as seguintes etapas:

1. **Carregamento do PDF:**  
   Utiliza `PyPDFLoader` para ler e dividir o PDF em páginas.

2. **Divisão em chunks:**  
   Usa `RecursiveCharacterTextSplitter` para dividir o texto das páginas em pedaços menores (chunks), facilitando o processamento.

3. **Geração de embeddings:**  
   Cada chunk é convertido em um vetor de embeddings usando o modelo `nomic-embed-text` via Ollama.

4. **Criação do banco vetorial:**  
   Os embeddings são armazenados em um banco de dados vetorial FAISS, permitindo buscas rápidas por similaridade.

5. **Configuração do RetrievalQA:**  
   O modelo LLM (`llama3`) é integrado ao banco vetorial via `RetrievalQA`, permitindo responder perguntas com base nos chunks mais relevantes.

6. **Execução de perguntas:**  
   O usuário faz uma pergunta, que é processada pelo pipeline e respondida com base no conteúdo do PDF.

---

## Execução do código

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

pdf_path = 'PPC_TSI_EaD.pdf'
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
pages = pages[:3]  # Processa as 3 primeiras páginas

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len,
)

texts = text_splitter.split_documents(pages)

db = FAISS.from_documents(texts, OllamaEmbeddings(model="nomic-embed-text"))

# Cria o modelo LLM e o retriever
model = OllamaLLM(model="llama3")
retriever = db.as_retriever(search_kwargs={"k": 5})

# Cria a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")

# Faz a pergunta
query = "Qual o nome do curso?"
result = qa_chain.invoke(query)
print(result)
```

---

## Personalização

- **Para processar mais páginas:**  
  Altere `pages = pages[:3]` para o número desejado ou remova o slice para processar todo o PDF.

- **Para fazer perguntas diferentes:**  
  Modifique o valor da variável `query`.

- **Para usar outros modelos:**  
  Altere o nome do modelo em `OllamaEmbeddings(model="...")` ou `OllamaLLM(model="...")` conforme os modelos disponíveis no seu Ollama.

---

## Referências

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.com/)
- [FAISS Documentation](https://faiss.ai/)
- [LangChain RetrievalQA](https://python.langchain.com/docs/use_cases/question_answering/)
