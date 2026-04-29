
import os
from pathlib import Path
try:
    import pypdf
except ImportError:
    pypdf = None

def extract_title_from_pdf(pdf_path):
    if not pypdf:
        return "pypdf not installed"
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        if len(reader.pages) > 0:
            text = reader.pages[0].extract_text()
            # Take the first few lines as a potential title
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
                return " ".join(lines[:3]) # First 3 lines
        return "Could not extract text"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    papers_dir = Path('c:/Users/Windows/PycharmProjects/NewAIScientist/papers')
    results = {}
    for pdf_file in papers_dir.glob('*.pdf'):
        if pdf_file.stat().st_size < 5000: # Skip small files (HTML placeholders)
            continue
        title = extract_title_from_pdf(pdf_file)
        results[pdf_file.name] = title
    
    import json
    with open('paper_titles_from_content.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Processed {len(results)} PDFs")
