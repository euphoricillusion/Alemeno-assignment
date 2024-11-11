import pickle

# Load the embeddings dictionary from the Pickle file
with open('document_embeddings.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)

# Access the embeddings and content
for file_path, data in embeddings_dict.items():
    print(f"File: {file_path}")
    print(f"Content Preview: {data['text'][:200]}")  # Preview first 200 characters
    print(f"Embeddings (first 10 values): {data['embedding'][:10]}")  # Display first 5 values
