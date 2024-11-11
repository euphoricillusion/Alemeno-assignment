from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from sentence_transformers import SentenceTransformer
import faiss
import os

# Initialize the model for creating embeddings (Sentence-Transformers)
model = SentenceTransformer('all-MiniLM-L6-v2')  # This is a lightweight model

# Function to load documents from a directory
def load_documents(doc_dir):
    documents = []
    for filename in os.listdir(doc_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(doc_dir, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:  # specify UTF-8 encoding
                text = file.read()
                documents.append({"text": text, "filename": filename})
    return documents

# Directory where your documents are stored (update this path)
doc_dir = r"c:\Users\user\OneDrive\Documents\Alemeno assignment"  # Folder containing your .txt files

# Load documents
documents = load_documents(doc_dir)

print(f"Loaded {len(documents)} documents.")
