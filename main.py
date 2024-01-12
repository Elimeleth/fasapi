from typing import Union
from pinecone_text.sparse import BM25Encoder
from pinecone_text.dense import SentenceTransformerEncoder

tokenizer_dense = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")

tokenizer_sparse = BM25Encoder.default()

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/dense/{text}")
def read_item(text: str):
    return tokenizer_dense.encode_queries(text)

@app.get("/sparse/{text}")
def read_item(text: str):
    return tokenizer_sparse.encode_queries(text)