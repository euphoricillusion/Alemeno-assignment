# Function to display extracted text from a file
def display_extracted_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the content of the file
            text = file.read()
            print(f"Displaying content from {file_path}:\n")
            # Display the first 500 characters to avoid flooding the console with large files
            print(text[:500])  # Adjust 500 to any number of characters you want to display
            print("\n... (more content below) ...\n")
            print(f"Full content saved in {file_path}")
    except Exception as e:
        print(f"Error reading the file {file_path}: {e}")

# List of extracted .txt files
pdf_files = [
    r"c:\Users\user\OneDrive\Documents\Alemeno assignment\alphabet_10k.pdf_extracted.txt",
    r"c:\Users\user\OneDrive\Documents\Alemeno assignment\tesla_10k.pdf_extracted.txt",
    r"c:\Users\user\OneDrive\Documents\Alemeno assignment\uber_10k.pdf_extracted.txt"
]

# Loop through each file and display content
for pdf_file in pdf_files:
    display_extracted_text(pdf_file)
