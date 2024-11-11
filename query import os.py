import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
import fitz  # PyMuPDF library for PDF extraction

# Initialize the model for creating embeddings (Sentence-Transformers)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model

# Function to load documents from a directory and convert PDFs to text using PyMuPDF
def load_documents_from_pdfs(doc_dir):
    documents = []
    for filename in os.listdir(doc_dir):
        if filename.endswith(".pdf"):  # Process only PDF files
            file_path = os.path.join(doc_dir, filename)
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text("text")  # Get text content of the page
                
                # Split the text into sentences (instead of paragraphs)
                sentences = text.split('. ')  # Basic sentence splitting by period and space
                for sentence in sentences:
                    if sentence.strip():  # Only add non-empty sentences
                        documents.append({"text": sentence.strip(), "filename": filename})
    return documents

# Directory where your PDFs are stored
doc_dir = r"c:\Users\user\OneDrive\Documents\Alemeno assignment"
documents = load_documents_from_pdfs(doc_dir)
print(f"Loaded {len(documents)} sentences from documents.")

# Step 1: Check if embeddings are already cached
def load_embeddings():
    if os.path.exists("embeddings.pkl"):
        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        return embeddings
    return None

def save_embeddings(embeddings):
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

# Step 2: Convert sentences to embeddings (Batch processing)
def create_embeddings(documents):
    sentences = [doc['text'] for doc in documents]
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=32)  # Batch processing
    return embeddings

# Step 3: Create or load the FAISS index
def create_faiss_index(embeddings):
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(embeddings_np)
    faiss.write_index(index, "sentence_index.faiss")
    return index

# Check if embeddings exist, otherwise create them
embeddings = load_embeddings()
if embeddings is None:
    embeddings = create_embeddings(documents)
    save_embeddings(embeddings)  # Cache embeddings to avoid recomputing

# Load or create the FAISS index
if os.path.exists("sentence_index.faiss"):
    index = faiss.read_index("sentence_index.faiss")
else:
    index = create_faiss_index(embeddings)

# Step 4: Real-time query search for dynamic retrieval
def search_query(query, top_k=50000):
    query_embedding = model.encode(query).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize query

    # Search for the most relevant sentences
    D, I = index.search(np.array([query_embedding]), k=top_k)

    # Extract and display relevant sentences
    print(f"\nQuery: {query}")
    relevant_sentences = []

    # Ensure we don't go out of range and extract relevant sentences
    for i, idx in enumerate(I[0]):
        if idx < len(documents):  # Ensure index is within range
            relevant_sentences.append(documents[idx]['text'])
        else:
            print(f"Warning: Index {idx} is out of range.")

    if relevant_sentences:
        # Process and clean the answer
        detailed_answer = process_answer(relevant_sentences)
        print("Answer:\n", detailed_answer)
    else:
        print("No relevant information found.")

def process_answer(relevant_sentences):
    """
    Process and combine relevant sentences into a cleaner, more readable answer.
    """
    # Join the sentences into one text block
    full_text = " ".join(relevant_sentences)
    
    # Remove excessive whitespace, special characters, etc.
    cleaned_text = re.sub(r'\s+', ' ', full_text)  # Remove extra spaces and line breaks
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,?!\s]', '', cleaned_text)  # Remove non-alphanumeric characters
    
    # Try to clean the output and make it more readable
    cleaned_text = cleaned_text.strip().replace('  ', ' ')  # Remove double spaces

    # If it's too short, possibly failed to get enough relevant data
    if len(cleaned_text) < 100:
        return "No detailed answer found."

    # Return the cleaned, readable version of the answer
    return cleaned_text

# Interactive chatbot loop
if __name__ == "__main__":
    print("Welcome to the Real-Time Document Search System!")
    while True:
        query = input("Enter a query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        search_query(query)
