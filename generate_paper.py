import os
import json
import textwrap
from fpdf import FPDF

class PaperPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Towards an AI Co-Scientist: Framework Evolution and RAG/CAG Optimization", align="C", ln=True)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
def safe_str(s):
    if not isinstance(s, str):
        return ""
    s = s.encode('latin-1', 'replace').decode('latin-1')
    
    # Manually break long words by inserting spaces, letting fpdf2 do word wrap natively
    words = s.split()
    safe_words = []
    for w in words:
        if len(w) > 40:
            # chunk the word into 40 char pieces
            safe_words.append(" ".join([w[i:i+40] for i in range(0, len(w), 40)]))
        else:
            safe_words.append(w)
            
    return " ".join(safe_words)

def safe_write_multiline(pdf, text, line_height, width=90):
    import textwrap
    lines = textwrap.wrap(text, width=width, replace_whitespace=False)
    for line in lines:
        try:
            pdf.cell(0, line_height, line, new_x="LMARGIN", new_y="NEXT")
        except TypeError:
            # Fallback for older fpdf versions
            pdf.cell(0, line_height, line, ln=1)

def generate_pdf(results_file="co_scientist_results.json"):
    pdf = PaperPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    
    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 15, "AI Co-Scientist: Advanced Multi-Agent Research System", align="C", ln=1)
    pdf.set_font("Helvetica", "I", 12)
    pdf.cell(0, 10, "A System Architecture and Performance Report", align="C", ln=1)
    pdf.ln(10)

    # 1. Abstract
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Abstract", ln=1)
    pdf.set_font("Helvetica", size=11)
    abstract = "This paper details the evolution of an AI Co-Scientist framework, designed to autonomously generate, critique, and evolve biological and scientific hypotheses. By combining dense Retrieval-Augmented Generation (RAG) with Context-Augmented Generation (CAG), the system effectively identifies novel research directions. Key innovations include True Divergent Thinking via explicit lateral ideation prompting, Cross-Domain Synthesis through knowledge graph extraction, and highly robust Citation Tracking using cryptographic hash-based IDs to eliminate hallucinated references."
    safe_write_multiline(pdf, abstract, 6)
    pdf.ln(5)
    
    # 2. Market Research
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Competitor Analysis & Positioning", ln=1)
    pdf.set_font("Helvetica", size=11)
    market = "Following an extensive market review, two primary paradigms emerged: 1. Autonomous Generators (e.g., Sakana.ai's The AI Scientist): Highly automated end-to-end paper generators that excel at volume but struggle with deep structural novelty. 2. Scientist-in-the-Loop Assistants (e.g., Google DeepMind's AI Co-Scientist framework): Focused on robust verification, multi-agent debate (Elo rating systems), and ethical safeguards. Our system evolves by integrating both: employing automated Phase 3 Lateral Ideation to match generative autonomy, while implementing a strict ID-based citation extraction."
    safe_write_multiline(pdf, market, 6)
    pdf.ln(5)
    
    # 3. Methodologies
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Core Methodologies", ln=1)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3.1 Improved Hybrid Context (RAG/CAG)", ln=1)
    pdf.set_font("Helvetica", size=11)
    meth1 = "In Phase 2, the legacy RAG pipeline was upgraded with Semantic Reranking. The retriever fetches candidate excerpts via cosine similarity, and an LLM acts as a judge to rerank the top k based on contextual relevance."
    safe_write_multiline(pdf, meth1, 6)
    
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3.2 True Divergent ideation & Graph Synthesis", ln=1)
    pdf.set_font("Helvetica", size=11)
    meth2 = "The EvolutionAgent was upgraded to support explicitly lateral thinking, prompting LLMs to borrow mechanisms from unrelated domains (e.g., astrophysics to oncology). The GraphAgent extracts entities to build an adjacency list, dynamically synthesizing bridging links between previously disconnected nodes."
    safe_write_multiline(pdf, meth2, 6)
    pdf.ln(5)

    # 4. Experimental Results
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "4. Experimental run output", ln=1)
    
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        goal = data.get("goal", {})
        pdf.set_font("Helvetica", "B", 12)
        title = str(goal.get("title", "Research Run")).replace('\n', ' ')
        safe_write_multiline(pdf, f"Goal: {title}", 8)
        pdf.set_font("Helvetica", size=11)
        
        hypotheses = data.get("hypotheses", [])
        if hypotheses:
            top_hyp = sorted(hypotheses, key=lambda x: x.get('elo_rating', 1200), reverse=True)
            for i, h in enumerate(top_hyp[:3]):
                htitle = str(h.get("title", "")).replace('\n', ' ').encode('ascii', 'replace').decode('ascii')
                elo = int(h.get("elo_rating", 1200))
                meth = str(h.get("generation_method", "")).replace('\n', ' ').encode('ascii', 'replace').decode('ascii')
                
                pdf.set_font("Helvetica", "B", 10)
                safe_write_multiline(pdf, f"Rank {i+1} [Elo {elo}] ({meth}): {htitle}", 6, width=80)
                
                pdf.set_font("Helvetica", size=10)
                desc = str(h.get("description", "")).replace('\n', ' ').encode('ascii', 'replace').decode('ascii')
                if len(desc) > 350: desc = desc[:347] + "..."
                safe_write_multiline(pdf, f"Description: {desc}", 5, width=95)
                pdf.ln(2)
        else:
            pdf.set_font("Helvetica", "I", 11)
            pdf.cell(0, 10, "[No simulation hypotheses found in results.]", ln=1)
    else:
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 10, "[No simulation results file found. Run app.py to generate experimental data.]", ln=1)
        
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "5. Conclusion", ln=1)
    pdf.set_font("Helvetica", size=11)
    conclusion = "The integration of multi-stage semantic filtering, ID-based citation tracking, and systematic divergent thinking significantly elevates the automated scientific discovery process. By prioritizing rigor and verifiable graph connections, the AI Co-Scientist serves as a powerful engine for accelerating multidisciplinary breakthroughs while maintaining scientific integrity."
    safe_write_multiline(pdf, conclusion, 6)

    # Save
    pdf.output("ai_co_scientist_paper_detailed.pdf")
    print("PDF successfully generated: ai_co_scientist_paper_detailed.pdf")

if __name__ == "__main__":
    generate_pdf()
