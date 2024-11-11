import pickle
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize the BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_embeddings(text):
    """
    Generate embeddings for the provided text using a pre-trained BERT model.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the hidden states to get a single embedding
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings

# **Define your file paths here**
file_paths = [
    r"c:\Users\user\OneDrive\Documents\Alemeno assignment\alphabet_10k.pdf_extracted.txt",
    r"c:\Users\user\OneDrive\Documents\Alemeno assignment\tesla_10k.pdf_extracted.txt",
    r"c:\Users\user\OneDrive\Documents\Alemeno assignment\uber_10k.pdf_extracted.txt"
]

# Dictionary to store embeddings along with text content
embeddings_dict = {}

# Loop through each file, read content, generate embeddings, and store in dictionary
for file_path in file_paths:
    try:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            embeddings = generate_embeddings(content)
            embeddings_dict[file_path] = {
                'text': content,
                'embedding': embeddings.numpy()  # Convert torch tensor to numpy
            }
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Save embeddings dictionary to a Pickle file
with open('document_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings_dict, f)

print("Embeddings stored successfully in 'document_embeddings.pkl'!")
