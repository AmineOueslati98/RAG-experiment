import os
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema.document import Document

import json
f = open('sample.json')
sample = json.load(f)


model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

text_chunks = ['Name:' + text for text in sample[0]['content'].split('Name:')[1:]]
clean_text = ' #$#$# '.join(text_chunks)


from langchain.text_splitter import CharacterTextSplitter


def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(    
        separator = "#$#$#",
        chunk_size = 500,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    return docs
text_chunks = get_text_chunks_langchain(clean_text)


vector_store = Chroma.from_documents(text_chunks, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/companies_cosine")

print("Vector Store Created.......")