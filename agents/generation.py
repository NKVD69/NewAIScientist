"""
agents/generation.py — GenerationAgent for hypothesis generation.

Responsible for:
- Generating initial hypotheses using LLM with Self-Refinement
- Building context-aware prompts from literature and RAG
- Simulated fallback generation when LLM is unavailable
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import config
from models.hypothesis import Hypothesis, ResearchGoal
from utils.llm import get_llm_completion, parse_json_response, ensure_str

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None


class GenerationAgent:
    """Generates initial hypotheses and explores research space"""
    
    def __init__(self, use_local_llm: bool = True):
        self.name = "Generation"
        self.generated_count = 0
        self.llm_client = None
        self.last_error = None
        self.cag_context = None  # Injected by CoScientist after literature search
        
        if use_local_llm and openai:
            try:
                self.llm_client = config.get_openai_client()
                logger.info("Generation Agent initialized with local LLM connection.")
            except Exception as e:
                logger.warning("Could not connect to local LLM: %s", e)
                self.llm_client = None
        elif use_local_llm and not openai:
            print("[WARN] `openai` library not found. Falling back to simulated generation.")

    async def generate_initial_hypotheses(self, 
                                        goal: ResearchGoal, 
                                        context_papers: List[Dict],
                                        count: int = 5,
                                        rag_context: List[Dict] = None) -> List[Hypothesis]:
        """Generate initial hypotheses using LLM with Self-Refinement"""
        print(f"💡 Generating {count} initial hypotheses with Self-Refinement...")
        
        if self.llm_client:
            try:
                # 1. Generate Drafts
                print("   ✍️ Generating Drafts...")
                draft_hypotheses = await self._generate_with_llm(goal, context_papers, count, rag_context)
                
                if not draft_hypotheses:
                    self.last_error = "LLM returned empty draft list"
                    return await self._generate_simulated(goal, count)

                # 2. Refine Drafts (Self-Correction)
                print("   🛡️ Critiquing and Refining Drafts...")
                refined_hypotheses = []
                for draft in draft_hypotheses:
                    refined = await self._refine_hypothesis(draft, goal)
                    refined_hypotheses.append(refined)
                
                self.generated_count += len(refined_hypotheses)
                return refined_hypotheses

            except Exception as e:
                import traceback
                self.last_error = f"LLM generation failed: {str(e)}\n{traceback.format_exc()}"
                print(f"⚠ {self.last_error}")

        if not self.last_error:
             self.last_error = "LLM client not initialized (check logs)"
        
        return await self._generate_simulated(goal, count)

    async def _refine_hypothesis(self, draft: Hypothesis, goal: ResearchGoal) -> Hypothesis:
        """Critique and refine a single hypothesis"""
        if not self.llm_client:
            return draft
            
        prompt = f"""
        Critique and refine the following scientific hypothesis to ensure it is rigorous, specific, and testable.
        
        Research Goal: {goal.title}
        
        Draft Hypothesis:
        Title: {draft.title}
        Description: {draft.description}
        Mechanism: {draft.mechanism}
        
        Identify 1 weakness (e.g. vague mechanism, lack of feasibility) and ANY improvements.
        Then rewrite the hypothesis in the SAME JSON format as the input, but improved.
        Output ONLY the JSON object.
        """
        
        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=True
            )
            content = response.choices[0].message.content.strip()
            
            # Robust parsing: remove markdown code blocks
            if content.startswith("```json"): content = content[7:]
            if content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            
            # Update draft with refined content
            draft.title = ensure_str(data.get("title", draft.title))
            draft.description = ensure_str(data.get("description", draft.description))
            draft.reasoning = ensure_str(data.get("reasoning", draft.reasoning))
            draft.mechanism = ensure_str(data.get("mechanism", draft.mechanism))
            draft.testable_predictions = data.get("testable_predictions", draft.testable_predictions)
            draft.limitations = data.get("limitations", draft.limitations)
            
            return draft
        except Exception as e:
            print(f"⚠ Refinement failed for '{draft.title}': {e}")
            return draft

    def _build_llm_prompt(self, goal: ResearchGoal, context_papers: List[Dict], count: int, rag_context: List[Dict] = None) -> str:
        """Build the LLM prompt for hypothesis generation"""
        
        literature_context = ""
        if context_papers:
            literature_context = "\n**Relevant Literature Context:**\n"
            for i, paper in enumerate(context_papers, 1):
                paper_id = hashlib.md5(paper.get('url', str(i)).encode()).hexdigest()[:8]
                literature_context += f"[{paper_id}] {paper['title']} ({paper.get('published', 'N/A')}): {paper.get('summary', '')[:800]}...\n"
        
        # Add RAG context if available (full-text chunks)
        rag_context_text = ""
        if rag_context:
            rag_context_text = "\n**Deep Literature Analysis (RAG):**\n"
            rag_context_text += "The following are the most relevant passages from full papers on this topic:\n\n"
            for i, chunk in enumerate(rag_context, 1):
                chunk_id = chunk.get('paper_id', str(i))[:8]
                rag_context_text += f"Excerpt {i} from '[{chunk_id}] {chunk.get('paper_title', 'Unknown')}':\n"
                rag_context_text += f"{chunk.get('text', '')[:1000]}...\n\n"
        
        # CAG: Inject Key Findings if available (Hybrid Context)
        key_findings = ""
        if self.cag_context:
            key_findings = f"""
**Contexte Scientifique Global (CAG):**
Utilisez ces découvertes clés issues de la littérature pour ancrer vos hypothèses dans la réalité actuelle du domaine :
{self.cag_context}
"""

        return f"""
Vous êtes un assistant de recherche IA expert. Votre tâche est de générer {count} hypothèses scientifiques novatrices et testables basées sur l'objectif de recherche et le contexte littéraire fournis.

**Objectif de Recherche :**
- **Titre :** {goal.title}
- **Domaine :** {goal.domain}
- **Description :** {goal.description}
- **Contraintes :** {', '.join(goal.constraints)}
- **Préférences :** {json.dumps(goal.preferences)}

{key_findings}

{literature_context}

{rag_context_text}

Veuillez générer {count} hypothèses distinctes. Pour chaque hypothèse, fournissez les informations suivantes dans un format JSON valide au sein d'un seul tableau JSON `[]`. **Toute l'argumentation et les descriptions doivent être en français.**

**Structure JSON pour chaque hypothèse :**
{{
  "title": "Un titre précis, technique et descriptif (ex: 'Inhibition de X via la voie Y pour Z').",
  "description": "Une explication approfondie de l'hypothèse. Soyez précis sur les cibles, les processus et les impacts attendus. Évitez les généralités.",
  "reasoning": "Détaillez le raisonnement logique et les données bibliographiques spécifiques qui ont permis de formuler cette hypothèse. Expliquez la connexion entre les preuves existantes et l'idée nouvelle.",
  "mechanism": "Décrivez précisément le mécanisme biochimique ou physique proposé. Comment les différentes composantes interagissent-elles ?",
  "testable_predictions": ["Liste de prédictions techniques et quantifiables.", "Prédiction 2", "..."],
  "cited_papers": ["Liste STRICte des identifiants des articles (ex: '[a1b2c3d4]', '[e5f6g7h8]') trouvés dans le contexte. N'inventez pas d'identifiants.", "..."],
  "grounding_evidence": ["Références précises issues du contexte ou principes physiques fondamentaux.", "Preuve 2", "..."],
  "limitations": ["Analyse critique des failles potentielles de l'hypothèse.", "Limitation 2", "..."]
}}

Assurez-vous que la sortie entière est un seul tableau JSON contenant {count} objets d'hypothèse. N'incluez aucun autre texte ou explication en dehors du tableau JSON.
"""

    async def _generate_with_llm(self, goal: ResearchGoal, context_papers: List[Dict], count: int, rag_context: List[Dict] = None) -> List[Hypothesis]:
        """Generate hypotheses using a local LLM."""
        prompt = self._build_llm_prompt(goal, context_papers, count, rag_context)
        
        try:
            response = await get_llm_completion(
                self.llm_client,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                json_mode=True
            )
        except Exception as e:
            print(f"⚠ LLM generation failed: {e}")
            return []
        
        content = response.choices[0].message.content
        
        try:
            data = parse_json_response(content)
            
            # Handle various JSON structures
            if isinstance(data, list):
                hypotheses_data = data
            elif isinstance(data, dict):
                list_values = [v for v in data.values() if isinstance(v, list)]
                if list_values:
                    hypotheses_data = list_values[0]
                else:
                    hypotheses_data = [data]
            else:
                raise ValueError("Unexpected JSON structure")

            hypotheses = []
            for item in hypotheses_data:
                cited_ids = item.get("cited_papers", [])
                
                # Reverse mapping: Find actual titles from context papers based on IDs
                cited_titles = []
                for cited_id in cited_ids:
                    clean_id = cited_id.strip("[]")
                    for p in context_papers:
                        p_id = hashlib.md5(p.get('url', '').encode()).hexdigest()[:8]
                        if clean_id == p_id:
                            cited_titles.append(f"{p['title']} ({p.get('published', '')})")
                            break
                            
                if not cited_titles and context_papers:
                    cited_titles = [f"{p['title']} ({p.get('published', '')})" for p in context_papers[:2]]
                
                grounding = item.get("grounding_evidence", [])
                if not grounding and cited_titles:
                    grounding = [f"Supported by: {t}" for t in cited_titles[:2]]
                
                hypotheses.append(Hypothesis(
                    title=ensure_str(item.get("title", "")),
                    description=ensure_str(item.get("description", "")),
                    reasoning=ensure_str(item.get("reasoning", "")),
                    mechanism=ensure_str(item.get("mechanism", "")),
                    testable_predictions=item.get("testable_predictions", []),
                    grounding_evidence=grounding,
                    limitations=item.get("limitations", []),
                    cited_papers=cited_titles,
                    generation_method="llm-generated"
                ))
            return hypotheses
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"⚠ Error parsing LLM response: {e}")
            print(f"  Raw response: {content}")
            return []

    async def _generate_simulated(self, goal: ResearchGoal, count: int) -> List[Hypothesis]:
        """Generate initial hypotheses using simulation (fallback)."""
        hypotheses = []
        strategies = [
            self._generate_from_literature,
            self._generate_from_assumptions,
            self._generate_from_analogies,
            self._generate_unconventional,
        ]
        
        for i in range(count):
            strategy = strategies[i % len(strategies)]
            hypothesis = await strategy(goal, i)
            hypotheses.append(hypothesis)
            self.generated_count += 1
        
        return hypotheses
    
    async def _generate_from_literature(self, goal: ResearchGoal, index: int) -> Hypothesis:
        """Simulate literature exploration-based generation"""
        h = Hypothesis(
            title=f"Hypothèse {index+1} (Basée sur la littérature) : Mécanisme de {goal.domain}",
            description=f"Cette hypothèse propose un nouveau mécanisme intégrant les recherches existantes sur {goal.domain}.",
            reasoning=f"Le raisonnement repose sur l'observation que les études récentes montrent une corrélation mais pas de causalité claire.",
            mechanism=f"S'appuyant sur la compréhension actuelle de {goal.domain}, ce mécanisme suggère une intégration interdisciplinaire.",
            generation_method="simulated-literature"
        )
        h.testable_predictions = [
            f"Prédiction 1 : Effet observable dans le contexte de {goal.domain}",
            f"Prédiction 2 : Conséquence mesurable en aval",
            f"Prédiction 3 : Changement de paramètre quantifiable"
        ]
        h.grounding_evidence = ["Analyse de la littérature existante", "Données expérimentales publiées"]
        h.cited_papers = h.grounding_evidence.copy()
        return h
    
    async def _generate_from_assumptions(self, goal: ResearchGoal, index: int) -> Hypothesis:
        """Generate from iterative assumptions"""
        h = Hypothesis(
            title=f"Hypothèse {index+1} (Basée sur les suppositions) : Nouvelle cible dans {goal.domain}",
            description=f"Cette hypothèse identifie une nouvelle cible thérapeutique potentielle dans {goal.domain}.",
            reasoning=f"Si l'on suppose que la cible Z est en fait un effet et non une cause, l'analyse logique pointe vers un régulateur amont.",
            mechanism="Chaîne logique : Si la supposition X tient → effet intermédiaire → résultat final",
            generation_method="simulated-assumption"
        )
        h.testable_predictions = [
            "Test 1 : Valider la supposition fondamentale",
            "Test 2 : Mesurer l'effet intermédiaire",
            "Test 3 : Confirmer le résultat final"
        ]
        h.grounding_evidence = ["Déduction logique", "Cadre analytique inter-domaines"]
        h.limitations = ["Suppose une progression linéaire qui pourrait être plus complexe"]
        return h
    
    async def _generate_from_analogies(self, goal: ResearchGoal, index: int) -> Hypothesis:
        """Generate using analogical reasoning"""
        h = Hypothesis(
            title=f"Hypothèse {index+1} (Basée sur l'analogie) : Mécanisme inter-domaines pour {goal.domain}",
            description=f"Analogue aux mécanismes dans des domaines liés, proposant une nouvelle application à {goal.domain}",
            mechanism="Le mécanisme découvert dans le domaine A peut s'appliquer au domaine B par similarité structurelle",
            generation_method="simulated-analogy"
        )
        h.testable_predictions = [
            "Prédiction d'analogie 1 : Manifestation spécifique au domaine",
            "Prédiction d'analogie 2 : Correspondance testable",
        ]
        h.limitations = ["Le raisonnement analogique peut ne pas être entièrement transférable"]
        return h
    
    async def _generate_unconventional(self, goal: ResearchGoal, index: int) -> Hypothesis:
        """Generate unconventional/out-of-box ideas"""
        h = Hypothesis(
            title=f"Hypothesis {index+1} (Unconventional): Divergent Mechanism in {goal.domain}",
            description=f"Explores unconventional directions in {goal.domain} research",
            mechanism="Moving away from established paradigms to explore underexplored mechanistic space",
            generation_method="simulated-unconventional"
        )
        h.testable_predictions = [
            "High-risk high-reward prediction 1",
            "Divergent experimental approach"
        ]
        h.limitations = [
            "Deviates from mainstream thinking",
            "May require novel experimental techniques"
        ]
        return h


__all__ = ["GenerationAgent"]
