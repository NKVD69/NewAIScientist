
from fpdf import FPDF
from datetime import datetime

class ScientificPaper(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, 'AI Co-Scientist: Autonomous Multi-Agent Framework for Scientific Discovery', 0, 0, 'R')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_draw_color(44, 62, 80)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(5)

    def sub_section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(52, 73, 94)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)

    def body_text(self, text):
        self.set_font('Times', '', 11)
        self.set_text_color(0, 0, 0)
        # Ensure only ASCII characters are used
        clean_text = text.encode('ascii', 'ignore').decode('ascii')
        self.multi_cell(0, 6, clean_text, align='J')
        self.ln(4)

def generate_detailed_paper():
    pdf = ScientificPaper()
    pdf.alias_nb_pages()
    
    # ---------------------------------------------------------
    # TITLE PAGE
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font('Arial', 'B', 22)
    pdf.set_text_color(44, 62, 80)
    pdf.multi_cell(0, 12, 'AI Co-Scientist: An Autonomous Multi-Agent Framework for Accelerating Hypotheses Generation and Literature Synthesis', 0, 'C')
    
    pdf.ln(20)
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, 'Technical Report v2.1', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Core Engineering Team', 0, 1, 'C')
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, 'Advanced Agentic Discovery Laboratory', 0, 1, 'C')
    
    pdf.ln(30)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Abstract', 0, 1, 'C')
    pdf.set_font('Times', 'I', 11)
    abstract = (
        "The rapid acceleration of scientific publishing has created a significant informational bottleneck, "
        "where the rate of data production far exceeds human cognitive capacity for synthesis. This report "
        "introduces NewAIScientist, an autonomous multi-agent framework designed to navigate, synthesize, "
        "and reason over the vast landscape of scientific literature. By leveraging an 'Agentic RAG' architecture "
        "combined with a hierarchical supervisor model, the system autonomously conducts iterative research cycles. "
        "Each cycle involves sophisticated literature exploration, self-refined hypothesis generation, peer-review "
        "simulation with Elo-based ranking, and evolutionary idea refinement. We demonstrate the system's "
        "capabilities through an automated use case in oncology, identifying novel therapeutic targets and "
        "drug repurposing opportunities with high scientific grounding."
    )
    pdf.multi_cell(0, 6, abstract, align='C')
    
    # ---------------------------------------------------------
    # 1. INTRODUCTION
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.section_title('1. Introduction')
    
    pdf.body_text(
        "The current paradigm of scientific research is increasingly defined by the 'Big Data' challenge. "
        "In the biomedical domain alone, thousands of papers are published daily, making it practically "
        "impossible for individual researchers to maintain a truly comprehensive overview of their field. "
        "This fragmentation leads to redundant efforts and missed opportunities for cross-disciplinary synthesis."
    )
    
    pdf.body_text(
        "NewAIScientist emerges as a solution to this scalability crisis. Inspired by the 'The AI Scientist' "
        "framework (Sakana.ai) and recent developments from Google DeepMind, our implementation focuses on "
        "creating a robust, autonomous 'Co-Scientist.' This refers to an AI partner capable of executing all "
        "phases of the discovery process-from literature mining to hypothesis evaluation-without requiring "
        "continuous human intervention."
    )
    
    pdf.body_text(
        "This report details the technical architecture of the NewAIScientist tool, including its agentic "
        "orchestration, RAG implementation, and the results of its application to high-impact research areas."
    )

    # ---------------------------------------------------------
    # 2. SYSTEM ARCHITECTURE
    # ---------------------------------------------------------
    pdf.section_title('2. System Architecture')
    
    pdf.body_text(
        "The core of NewAIScientist is a modular multi-agent system governed by a supervisor. This architecture "
        "allows for specialized agents to handle distinct phases of the scientific workflow while maintaining "
        "a global context memory."
    )
    
    pdf.sub_section_title('2.1 Supervisor & Task Framework')
    pdf.body_text(
        "The SupervisorAgent acts as the central orchestrator. It manages a priority-based task queue, ensuring "
        "that dependencies (e.g., literature search before generation) are respected. The orchestration utilizes "
        "asynchronous execution to handle long-running LLM calls and PDF processing concurrently where possible."
    )
    
    pdf.sub_section_title('2.2 Literature Agent & Agentic RAG')
    pdf.body_text(
        "Unlike traditional RAG systems that rely on static document indices, our Literature Agent implements "
        "an 'Agentic RAG' pipeline. This involves:\n"
        "- Iterative Querying: The agent analyzes initial results to identify information gaps and "
        "refines its search queries dynamically.\n"
        "- Full-Text Acquisition: The system automatically pulls full PDF manuscripts from ArXiv and PubMed, "
        "ensuring that the reasoning is based on detailed data rather than just abstracts.\n"
        "- Semantic Chunking: Documents are split into semi-overlapping semantic chunks, which are "
        "indexed in a local ChromaDB vector store for high-precision retrieval."
    )

    # ---------------------------------------------------------
    # 3. THE DISCOVERY WORKFLOW
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.section_title('3. The Discovery Workflow')
    
    pdf.body_text(
        "The discovery process follows a strictly defined lifecycle for every generated idea, ensuring "
        "rigorous scientific validation."
    )
    
    pdf.sub_section_title('3.1 Hypothesis Generation with Self-Refinement')
    pdf.body_text(
        "The GenerationAgent produces initial hypotheses grounded in the retrieved literature. Crucially, it "
        "employs a 'Self-Refinement' loop: the agent generates a draft, critiques its own work for logical "
        "consistency or lack of evidence, and then submits an improved 'grounded' version. This drastically "
        "reduces hallucinations and ensures technical feasibility."
    )
    
    pdf.sub_section_title('3.2 Reflection & Peer Review')
    pdf.body_text(
        "Hypotheses are subjected to an automated peer review conducted by the ReflectionAgent. Each idea is "
        "scored across four primary dimensions:\n"
        "- Correctness (0-1): Logical consistency and grounding in known biological/physical laws.\n"
        "- Novelty (0-1): Distance from the existing state-of-the-art found in the literature.\n"
        "- Testability (0-1): The degree to which the hypothesis can be verified via current "
        "experimental techniques.\n"
        "- Quality (0-1): An aggregate score reflecting the overall potential for publication."
    )
    
    pdf.sub_section_title('3.3 Tournament Ranking & Elo System')
    pdf.body_text(
        "To identify truly superior ideas, the system conducts pairwise tournaments between hypotheses. The "
        "RankingAgent facilitates scientific 'debates' where agents argue the merits of one hypothesis over "
        "another. Resulting victories update global Elo ratings, allowing the most promising directions to emerge "
        "organically from the pool."
    )

    # ---------------------------------------------------------
    # 4. CASE STUDY: DRUG REPURPOSING FOR AML
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.section_title('4. Case Study: Drug Repurposing for AML')
    
    pdf.body_text(
        "To demonstrate the system's power, we tasked it with identifying novel therapeutic strategies for "
        "Acute Myeloid Leukemia (AML). The objective was to find FDA-approved drugs capable of inducing apoptosis "
        "in leukemic clones through unconventional pathways."
    )
    
    pdf.body_text(
        "Execution Phases:\n"
        "1. Exploration: The system retrieved and analyzed 19 full-text manuscripts focusing on AML mutation "
        "profiles and endoplasmic reticulum (ER) stress.\n"
        "2. Mechanism Discovery: The agents identified the IRE1alpha (Inositol-requiring enzyme 1) pathway as "
        "a critical, underexploited vulnerability. The system hypothesized that IRE1alpha-mediated degradation of p53 "
        "mRNA contributes to drug resistance in certain AML subtypes.\n"
        "3. Proposed Intervention: The framework suggested that small-molecule inhibitors of IRE1alpha endoribonuclease "
        "activity could restore p53 function, potentially sensitizing resistant cells to standard chemotherapy."
    )
    
    pdf.body_text(
        "Quantitative Results:\n"
        "- Initial Hypothesis Pool: 5 candidates.\n"
        "- Refinement Cycles: 3 iterations.\n"
        "- Winning Elo Score: 1456 (Standardized relative to 1200 base).\n"
        "- Reviewer Consensus: 'High Novelty' and 'High Testability'."
    )

    # ---------------------------------------------------------
    # 5. TECHNICAL IMPLEMENTATION
    # ---------------------------------------------------------
    pdf.section_title('5. Technical Implementation')
    pdf.body_text(
        "NewAIScientist is implemented using a modern Python stack optimized for local execution:\n"
        "- Core Language: Python 3.9+ with asyncio for task concurrency.\n"
        "- Frontend: Streamlit dashboard with real-time progress visualization.\n"
        "- Database: ChromaDB for local vector storage and persistent context memory.\n"
        "- PDF Processing: pypdf for text extraction and cleaning.\n"
        "- LLM Integration: Compatible with any OpenAI-style API, specifically optimized for local "
        "backends like LM Studio and Ollama to ensure data privacy."
    )

    # ---------------------------------------------------------
    # 6. FUTURE DIRECTIONS
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.section_title('6. Future Directions')
    pdf.body_text(
        "While NewAIScientist represents a significant step toward autonomous discovery, several key areas for "
        "future expansion have been identified to further enhance its capabilities:"
    )
    
    pdf.body_text(
        "- Multi-modal Support: The current system only processes text. Integrating image and figure extraction "
        "from PDFs could improve grounding by allowing the agents to reason over experimental data, charts, "
        "and biological diagrams directly.\n"
        "- Distributed Processing: The supervisor could be extended to run agent tasks across multiple threads "
        "or distributed machines. This would allow for significantly larger research goals involving thousands "
        "of papers to be processed in parallel.\n"
        "- Better Citation Tracking: While the system currently tracks primary sources, implementing a formal "
        "citation index and graph analysis could further strengthen the meta-analysis by identifying the most "
        "influential 'node' papers in a specific field.\n"
        "- Extensible Agent Registry: Making it easier to plug in custom agents for specific scientific "
        "domains (e.g., a 'ChemistryAgent' for safety and toxicity checks, or a 'BioinformaticsAgent' for "
        "sequence analysis) would make the framework truly universal."
    )

    # ---------------------------------------------------------
    # 7. CONCLUSION
    # ---------------------------------------------------------
    pdf.section_title('7. Conclusion')
    pdf.body_text(
        "The NewAIScientist framework demonstrates that autonomous discovery is no longer a distant possibility "
        "but a present reality. By delegating the heavy lifting of literature synthesis and preliminary reasoning "
        "to a specialized multi-agent system, researchers can transcend traditional informational bottlenecks.\n\n"
        "The roadmap for NewAIScientist leads toward a true closed-loop discovery system where hypotheses are "
        "not only generated but also autonomously verified in the laboratory through API-controlled liquid "
        "handling systems and real-time sensor feedback."
    )

    output_path = 'ai_co_scientist_paper_detailed.pdf'
    pdf.output(output_path)
    print(f"Detailed paper generated successfully: {output_path}")

if __name__ == "__main__":
    generate_detailed_paper()
