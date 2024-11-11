import PyPDF2

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()  # Extract text from each page
    return text

# Path to your PDFs (Make sure these files are in the same folder or provide the full path)
pdf_files = [r"c:\Users\user\OneDrive\Documents\Alemeno assignment\alphabet_10k.pdf",
             r"c:\Users\user\OneDrive\Documents\Alemeno assignment\tesla_10k.pdf",
             r"c:\Users\user\OneDrive\Documents\Alemeno assignment\uber_10k.pdf"]

# Loop through all the files and extract text
for pdf_file in pdf_files:
    print(f"Extracting text from {pdf_file}...")
    text = extract_text_from_pdf(pdf_file)
    # You can save the extracted text to a file or print it out for now
    with open(f"{pdf_file}_extracted.txt", "w", encoding="utf-8") as output_file:
        output_file.write(text)
    print(f"Text extraction complete for {pdf_file}. Text saved to {pdf_file}_extracted.txt\n")
