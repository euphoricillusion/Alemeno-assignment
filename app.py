import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize conversational model (local GPT model)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize embedding model for document retrieval
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents from a directory
def load_documents(doc_dir):
    documents = []
    for filename in os.listdir(doc_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(doc_dir, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                documents.append({"text": text, "filename": filename})
    return documents

# Directory where documents are stored
doc_dir = r"c:\Users\user\OneDrive\Documents\Alemeno assignment"
documents = load_documents(doc_dir)
print(f"Loaded {len(documents)} documents.")

# Generate document embeddings
def generate_document_embeddings(documents):
    embeddings = []
    for doc in documents:
        embedding = embedding_model.encode(doc['text'])
        embeddings.append(embedding)
    return np.array(embeddings).astype('float32')

# Create FAISS index
embeddings = generate_document_embeddings(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Retrieve relevant document content
def retrieve_documents(query, top_k=2, threshold=0.7):
    query_embedding = embedding_model.encode(query).astype('float32')
    distances, indices = index.search(np.array([query_embedding]), k=top_k)
    
    # Filter documents based on similarity threshold
    relevant_docs = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i] < threshold:
            relevant_docs.append(documents[idx]['text'])
    
    return relevant_docs if relevant_docs else None

# Generate conversational response with document context
def generate_response(user_input, history=[], top_k=2):
    # Step 1: Retrieve relevant documents based on user query
    retrieved_docs = retrieve_documents(user_input, top_k=top_k)

    # Step 2: Prepare prompt based on retrieved content or default to a general response
    if retrieved_docs:
        context = " ".join(retrieved_docs)[:1500]  # Limit context length
        prompt = f"Context: {context}\n\nUser: {user_input}\nBot:"
    else:
        prompt = f"User: {user_input}\nBot:"  # General response if no relevant documents found

    # Encode input and generate response
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    bot_output = model.generate(
        input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id, 
        temperature=0.7, top_p=0.9, top_k=50, no_repeat_ngram_size=2
    )
    response = tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response

# Interactive chatbot loop
def start_chat():
    print("Welcome to the Offline Chatbot. Type 'exit' to stop.")
    history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = generate_response(user_input, history)
        print(f"Bot: {response}")

if __name__ == "__main__":
    start_chat()
