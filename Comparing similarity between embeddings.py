from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embeddings for a document
def generate_embeddings(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    sentences = content.split('\n')  # Split the content by new lines
    embeddings = model.encode(sentences)  # Generate embeddings for the sentences
    return embeddings

# Paths to your extracted text files
file_paths = [
    r"c:\Users\user\OneDrive\Documents\Alemeno assignment\alphabet_10k.pdf_extracted.txt",
    r"c:\Users\user\OneDrive\Documents\Alemeno assignment\tesla_10k.pdf_extracted.txt",
    r"c:\Users\user\OneDrive\Documents\Alemeno assignment\uber_10k.pdf_extracted.txt"
]

# Generate embeddings for each document
alphabet_embeddings = generate_embeddings(file_paths[0])
tesla_embeddings = generate_embeddings(file_paths[1])
uber_embeddings = generate_embeddings(file_paths[2])

# Compute cosine similarity between the documents
similarity_alphabet_tesla = cosine_similarity([alphabet_embeddings.mean(axis=0)], [tesla_embeddings.mean(axis=0)])
similarity_alphabet_uber = cosine_similarity([alphabet_embeddings.mean(axis=0)], [uber_embeddings.mean(axis=0)])
similarity_tesla_uber = cosine_similarity([tesla_embeddings.mean(axis=0)], [uber_embeddings.mean(axis=0)])

# Display the similarities
print(f"Cosine Similarity between Alphabet and Tesla: {similarity_alphabet_tesla[0][0]:.4f}")
print(f"Cosine Similarity between Alphabet and Uber: {similarity_alphabet_uber[0][0]:.4f}")
print(f"Cosine Similarity between Tesla and Uber: {similarity_tesla_uber[0][0]:.4f}")
print(f"Alphabet Embeddings: {alphabet_embeddings[:10]}")  # Print first 10 embedding values
print(f"Tesla Embeddings: {tesla_embeddings[:10]}")
print(f"Uber Embeddings: {uber_embeddings[:10]}")

# Debugging statements
print("Alphabet Embeddings Generated:", len(alphabet_embeddings))
print("Tesla Embeddings Generated:", len(tesla_embeddings))
print("Uber Embeddings Generated:", len(uber_embeddings))
print("First few sentences of Alphabet document:", open(file_paths[0], 'r', encoding='utf-8').read().split('\n')[:5])

print("Loading pre-trained model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")
