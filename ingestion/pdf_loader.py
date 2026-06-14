import fitz  # PyMuPDF
from pathlib import Path


# Folder Paths
RAW_PDF_DIR = Path("data/raw_pdfs")
EXTRACTED_DIR = Path("data/extracted")

# Create extracted folder if it doesn't exist
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """

    text = ""

    try:
        pdf_document = fitz.open(pdf_path)

        total_pages = len(pdf_document)
        print(f"Pages: {total_pages}")

        for page_num in range(total_pages):
            page = pdf_document[page_num]

            # Extract text from page
            page_text = page.get_text("text")

            if page_text:
                text += page_text + "\n"

        pdf_document.close()

    except Exception as e:
        print(f"Error reading {pdf_path.name}: {e}")

    return text


def process_all_pdfs():
    """
    Process all PDFs in raw_pdfs folder.
    """

    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/raw_pdfs/")
        return

    print(f"\nFound {len(pdf_files)} PDF file(s)\n")

    for pdf_file in pdf_files:

        print("=" * 60)
        print(f"Processing: {pdf_file.name}")

        extracted_text = extract_text_from_pdf(pdf_file)

        # Word count
        word_count = len(extracted_text.split())

        # Character count
        char_count = len(extracted_text)

        print(f"Words      : {word_count}")
        print(f"Characters : {char_count}")

        # Warning for empty extraction
        if word_count == 0:
            print("WARNING: No text extracted!")
            print("Possible causes:")
            print("- Scanned PDF")
            print("- Image-only PDF")
            print("- Protected PDF")

        output_file = EXTRACTED_DIR / f"{pdf_file.stem}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        print(f"Saved TXT  : {output_file.name}")

    print("\n" + "=" * 60)
    print("PDF extraction completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    process_all_pdfs()