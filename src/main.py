import json
import os
from datetime import datetime
from typing import Dict, List
import PyPDF2

from src.persona_analyzer import PersonaAnalyzer, PersonaProfile
from src.job_parser import JobParser
from src.relevance_scorer import RelevanceScorer
from src.content_refiner import ContentRefiner

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extracts text from a PDF file, keeping track of page numbers.
    For simplicity, treats each page as a single "section".
    """
    sections = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                sections.append({
                    "document": os.path.basename(pdf_path),
                    "page_number": i + 1,
                    "section_title": f"Page {i+1}", # Simplified title
                    "text": page.extract_text() or ""
                })
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return sections

def process_document_collection(input_data: Dict, collection_path: str) -> Dict:
    """
    Main processing pipeline for a collection of documents.
    """
    # 1. Initialize tools
    persona_analyzer = PersonaAnalyzer()
    job_parser = JobParser()
    content_refiner = ContentRefiner()

    # 2. Analyze Persona and Job
    persona_profile = persona_analyzer.analyze_persona_dict(input_data['persona'])
    job_details = job_parser.parse_job(input_data['job_to_be_done']['task'])

    # 3. Initialize Relevance Scorer
    scorer = RelevanceScorer(persona_profile, job_details)

    # 4. Process all documents to extract sections
    all_sections = []
    pdf_dir = os.path.join(collection_path, "PDFs")
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found at {pdf_dir}")

    for doc in input_data['documents']:
        pdf_path = os.path.join(pdf_dir, doc['filename'])
        if os.path.exists(pdf_path):
            all_sections.extend(extract_text_from_pdf(pdf_path))
        else:
            print(f"Warning: PDF file not found at {pdf_path}")

    # 5. Score all extracted sections
    section_texts = [section['text'] for section in all_sections]
    scores = scorer.score_sections(section_texts)

    for i, section in enumerate(all_sections):
        section['score'] = scores[i]

    # 6. Rank sections and select the top ones
    all_sections.sort(key=lambda x: x['score'], reverse=True)
    top_sections = all_sections[:5] # Top 5 sections

    # 7. Refine content for subsection analysis
    subsection_analysis = []
    for section in top_sections:
        refined_text = content_refiner.refine_text(section['text'])
        subsection_analysis.append({
            "document": section['document'],
            "refined_text": refined_text,
            "page_number": section['page_number']
        })

    # 8. Format the final output
    output = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in input_data['documents']],
            "persona": persona_profile.role,
            "job_to_be_done": job_details['original_text'],
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": s['document'],
                "section_title": s['section_title'],
                "importance_rank": i + 1,
                "page_number": s['page_number']
            } for i, s in enumerate(top_sections)
        ],
        "subsection_analysis": subsection_analysis
    }

    return output

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/main.py <path_to_collection_directory>")
        sys.exit(1)

    collection_path = sys.argv[1]
    input_json_path = os.path.join(collection_path, 'challenge1b_input.json')
    output_json_path = os.path.join(collection_path, 'challenge1b_output.json')

    try:
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)

        final_output = process_document_collection(input_data, collection_path)

        with open(output_json_path, 'w') as f:
            json.dump(final_output, f, indent=4)

        print(f"Processing complete. Output saved to {output_json_path}")

    except FileNotFoundError:
        print(f"Error: Input file or PDF directory not found in {collection_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
