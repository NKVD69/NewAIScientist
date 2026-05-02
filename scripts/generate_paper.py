import asyncio
import json
import os
import unicodedata

from fpdf import FPDF

from co_scientist import _get_llm_completion, _parse_json_response, config


def _sanitize_for_pdf(text: str) -> str:
    """
    Convert unicode text to a PDF-safe string, preserving as much content as possible.
    Strategy:
    1. Try latin-1 directly (fast path for pure ASCII/Western content)
    2. NFKD normalize (é → e + combining accent, then drop combining chars) → better than direct encode
    3. Map common scientific symbols to ASCII approximations
    4. Last resort: replace with '?'
    """
    if not text:
        return ""

    # Common scientific symbol replacements
    SYMBOL_MAP = {
        '→': '->', '←': '<-', '↑': '^', '↓': 'v', '↔': '<->',
        '±': '+/-', '≈': '~', '≤': '<=', '≥': '>=', '≠': '!=',
        '×': 'x', '÷': '/', '∞': 'inf', '∑': 'sum', '∏': 'prod',
        '∫': 'int', '∂': 'd', '∇': 'nabla', '√': 'sqrt',
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'λ': 'lambda', 'μ': 'mu', 'π': 'pi',
        'σ': 'sigma', 'τ': 'tau', 'φ': 'phi', 'ω': 'omega',
        'Δ': 'Delta', 'Σ': 'Sigma', 'Ω': 'Omega', 'Γ': 'Gamma',
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '--', '\u2026': '...',
    }

    for char, replacement in SYMBOL_MAP.items():
        text = text.replace(char, replacement)

    try:
        text.encode('latin-1')
        return text
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # NFKD decomposition: strips combining diacritics (é → e)
    normalized = unicodedata.normalize('NFKD', text)
    try:
        normalized.encode('latin-1')
        return normalized
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    return normalized.encode('latin-1', errors='replace').decode('latin-1')


class PaperPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Towards an AI Co-Scientist: Agentic Paper Generation", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def safe_write_multiline(pdf, text, line_height, width=90):
    """Write text to PDF with safe Unicode handling."""
    sanitized = _sanitize_for_pdf(text)
    try:
        pdf.multi_cell(0, line_height, sanitized, new_x="LMARGIN", new_y="NEXT")
    except Exception:
        import textwrap
        for line in textwrap.wrap(sanitized, width=width):
            try:
                pdf.cell(0, line_height, line, new_x="LMARGIN", new_y="NEXT")
            except Exception:
                pass

async def draft_paper(llm_client, results_data):
    print("Drafting paper with LLM...")
    goal = results_data.get("goal", {})
    hypotheses = results_data.get("hypotheses", [])
    top_hyp = sorted(hypotheses, key=lambda x: x.get('elo_rating', 1200), reverse=True)[0] if hypotheses else {}

    prompt = f"""
    You are an expert AI Scientist. Write a full scientific manuscript based on the following research session.

    Research Goal: {goal.get('title', '')}
    Domain: {goal.get('domain', '')}
    Description: {goal.get('description', '')}

    Top Hypothesis Generated: {top_hyp.get('title', '')}
    Mechanism: {top_hyp.get('mechanism', '')}
    Experimental Results: {top_hyp.get('experimental_results', 'No experiments run.')}

    Write the paper in JSON format with exactly these fields: "abstract", "introduction", "methods", "results", "conclusion".
    Return ONLY the JSON. Write it in English and keep it rigorous.
    """
    response = await asyncio.wait_for(_get_llm_completion(llm_client, messages=[{"role": "user", "content": prompt}], json_mode=True), timeout=60)
    return _parse_json_response(response.choices[0].message.content)

async def peer_review_paper(llm_client, paper_draft):
    print("Performing Automated Peer Review...")
    prompt = f"""
    You are a harsh NeurIPS/Nature reviewer. Review the following paper draft:

    Abstract: {paper_draft.get('abstract', '')}
    Methods: {paper_draft.get('methods', '')}
    Results: {paper_draft.get('results', '')}
    Conclusion: {paper_draft.get('conclusion', '')}

    Evaluate the paper on Soundness, Presentation, and Contribution.
    Return a JSON object with:
    - "soundness_score" (0-10)
    - "presentation_score" (0-10)
    - "contribution_score" (0-10)
    - "review_feedback" (concise constructive criticism)
    """
    response = await asyncio.wait_for(_get_llm_completion(llm_client, messages=[{"role": "user", "content": prompt}], json_mode=True), timeout=60)
    return _parse_json_response(response.choices[0].message.content)

async def refine_paper(llm_client, paper_draft, review):
    print("Refining paper based on Peer Review...")
    prompt = f"""
    You are an AI Scientist revising a paper based on reviewer feedback.

    Original Abstract: {paper_draft.get('abstract', '')}
    Original Results: {paper_draft.get('results', '')}
    Original Conclusion: {paper_draft.get('conclusion', '')}

    Reviewer Feedback: {review.get('review_feedback', '')}

    Improve the paper to address the feedback.
    Return the revised paper in JSON format with fields: "abstract", "introduction", "methods", "results", "conclusion".
    Return ONLY JSON.
    """
    response = await asyncio.wait_for(_get_llm_completion(llm_client, messages=[{"role": "user", "content": prompt}], json_mode=True), timeout=60)
    return _parse_json_response(response.choices[0].message.content)

async def generate_pdf_async(results_file="co_scientist_results.json"):
    try:
        llm_client = config.get_openai_client()
    except Exception as e:
        print(f"Could not connect to LLM: {e}")
        return

    if not os.path.exists(results_file):
        print("Results file not found. Run app.py first.")
        return

    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    try:
        draft = await draft_paper(llm_client, data)
        review = await peer_review_paper(llm_client, draft)
        print(f"Peer Review Scores - Soundness: {review.get('soundness_score')}, Presentation: {review.get('presentation_score')}")
        refined = await refine_paper(llm_client, draft, review)
    except Exception as e:
        print(f"Error during LLM paper generation: {e}")
        return

    # Generate PDF
    pdf = PaperPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    pdf.set_font("Helvetica", "B", 18)
    goal_title = data.get('goal', {}).get('title', 'AI Generated Paper')
    safe_write_multiline(pdf, goal_title, 8, width=70)
    pdf.ln(10)

    sections = [
        ("1. Abstract", refined.get("abstract", "")),
        ("2. Introduction", refined.get("introduction", "")),
        ("3. Methods", refined.get("methods", "")),
        ("4. Results", refined.get("results", "")),
        ("5. Conclusion", refined.get("conclusion", ""))
    ]

    for title, text in sections:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, title, ln=1)
        pdf.set_font("Helvetica", size=11)
        safe_write_multiline(pdf, text, 6)
        pdf.ln(5)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Appendix: Peer Review Feedback Automatically Addressed", ln=1)
    pdf.set_font("Helvetica", "I", 10)
    safe_write_multiline(pdf, str(review.get("review_feedback", "")), 5)

    pdf.output("ai_co_scientist_paper_detailed.pdf")
    print("PDF successfully generated: ai_co_scientist_paper_detailed.pdf")

def generate_pdf(results_file="co_scientist_results.json"):
    asyncio.run(generate_pdf_async(results_file))

if __name__ == "__main__":
    generate_pdf()
