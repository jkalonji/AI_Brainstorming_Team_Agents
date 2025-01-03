import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch

# Read the knowledge base from the file 'lesson'
with open('lesson.txt', 'r', encoding='utf-8') as f:
    knowledge_base = f.read()

# Split the knowledge base into chunks using CharacterTextSplitter
text_splitter = CharacterTextSplitter(separator='\n\n', chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_text(knowledge_base)

# Load the embedding model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Create FAISS vector store
docsearch = FAISS.from_texts(chunks, embeddings)

# Function to get the most similar chunk
def get_most_similar_chunk(query, docsearch):
    docs = docsearch.similarity_search(query, k=1)
    return docs[0].page_content

# Example usage
query = "How to load the Boston Housing dataset?"
answer = get_most_similar_chunk(query, docsearch)
print(answer)