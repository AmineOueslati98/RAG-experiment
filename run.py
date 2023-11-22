from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from io import BytesIO
from langchain.document_loaders import PyPDFLoader


local_llm = "zephyr-7b-beta.Q5_K_S.gguf"

config = {
'max_new_tokens': 1024,
'repetition_penalty': 1.1,
'temperature': 0.1,
'top_k': 50,
'top_p': 0.9,
'stream': True,
'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model=local_llm,
    model_type="mistral",
    context_length=2048,
    lib="avx2", #for CPU use
    **config
)

print("LLM Initialized...")


prompt_template = """You are a truthful, helpful research assistant called PUMA (short for Processing Unit for Multilayered Analysis), working for a startup named Valuer. You will be provided with a question and a list of results that are semantically relevant to the query mentioned as a context, from the Valuer database. Your job is to answer the question as truthfully and concisely as possible. Please separate the response into natural paragraphs, and also other formats, such as lists, tables, etc. whenever useful.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
load_vector_store = Chroma(persist_directory="stores/companies_cosine", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k":1})
query = "what do you know about Digipay's unique selling point?"
semantic_search = retriever.get_relevant_documents(query)
print(semantic_search)


chain_type_kwargs = {"prompt": prompt}

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents = True,
    chain_type_kwargs= chain_type_kwargs,
    verbose=True
)

response = qa(query)

print(response)

