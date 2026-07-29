"""
agents/generation.py — GenerationAgent for hypothesis generation.

Responsible for:
- Generating initial hypotheses using LLM with Self-Refinement
- Building context-aware prompts from literature and RAG
- Simulated fallback generation when LLM is unavailable
"""

from __future__ import annotations

import hashlib
import json
import logging

from models.hypothesis import Claim, Evidence, Hypothesis, ResearchGoal
from utils.llm import ensure_str, get_llm_completion, parse_json_response

from .base import BaseAgent

logger = logging.getLogger(__name__)


class GenerationAgent(BaseAgent):
    """Generates initial hypotheses and explores research space"""

    name = "Generation"

    def __init__(self, use_local_llm: bool = True):
        super().__init__(use_local_llm=use_local_llm)
        self.generated_count = 0
        self.last_error = None
        self.cag_context = None  # Injected by CoScientist after literature search

    async def generate_initial_hypotheses(self,
                                        goal: ResearchGoal,
                                        context_papers: list[dict],
                                        count: int = 5,
                                        rag_context: list[dict] = None) -> list[Hypothesis]:
        """Generate initial hypotheses using LLM with Self-Refinement"""
        logger.info("Generating %d initial hypotheses with self-refinement...", count)

        if self.llm_client:
            try:
                # 1. Generate Drafts
                logger.info("Generating drafts...")
                draft_hypotheses = await self._generate_with_llm(goal, context_papers, count, rag_context)

                if not draft_hypotheses:
                    self.last_error = "LLM returned empty draft list"
                    return await self._generate_simulated(goal, count)

                # 2. Refine Drafts (Self-Correction)
                logger.info("Critiquing and refining drafts...")
                refined_hypotheses = []
                for draft in draft_hypotheses:
                    refined = await self._refine_hypothesis(draft, goal)
                    refined_hypotheses.append(refined)

                self.generated_count += len(refined_hypotheses)
                return refined_hypotheses

            except Exception as e:
                import traceback
                self.last_error = f"LLM generation failed: {str(e)}\n{traceback.format_exc()}"
                logger.warning(self.last_error)

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
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
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
            logger.warning("Refinement failed for '%s': %s", draft.title, e)
            return draft

    def _build_llm_prompt(self, goal: ResearchGoal, context_papers: list[dict], count: int, rag_context: list[dict] = None) -> str:
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
  "claims": [
    {{
      "statement": "Une affirmation atomique et falsifiable (une seule assertion par entrée).",
      "confidence": 0.7,
      "evidence": [
        {{"text": "Passage ou fait qui étaye cette affirmation précise.",
         "source_type": "rag|citation|prior",
         "source_ref": "L'identifiant [a1b2c3d4] de l'article, ou '' si connaissance générale.",
         "polarity": 1,
         "confidence": 0.6}}
      ]
    }}
  ],
  "grounding_evidence": ["Références précises issues du contexte ou principes physiques fondamentaux.", "Preuve 2", "..."],
  "limitations": ["Analyse critique des failles potentielles de l'hypothèse.", "Limitation 2", "..."]
}}

Assurez-vous que la sortie entière est un seul tableau JSON contenant {count} objets d'hypothèse. N'incluez aucun autre texte ou explication en dehors du tableau JSON.
"""

    async def _generate_with_llm(self, goal: ResearchGoal, context_papers: list[dict], count: int, rag_context: list[dict] = None) -> list[Hypothesis]:
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
            logger.warning("LLM generation failed: %s", e)
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
                    claims=self._build_claims(item, rag_context),
                    generation_method="llm-generated"
                ))
            return hypotheses
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning("Error parsing LLM response: %s", e)
            logger.debug("Raw response: %s", content)
            return []

    @staticmethod
    def _build_claims(item: dict, rag_context: list[dict] | None = None) -> list[Claim]:
        """Turn the LLM's ``claims`` block into structured Claim/Evidence objects.

        ``Claim`` and ``Evidence`` were defined in ``models/hypothesis.py`` with
        polarity, per-item confidence and a ``source_ref`` for provenance — and
        were never instantiated anywhere in production. This is that
        instantiation. It is what makes partial refutation and
        claim-to-chunk provenance possible: without it a hypothesis is an
        undifferentiated blob of prose and can only be accepted or rejected
        whole.

        Degrades gracefully: a model that ignores the ``claims`` key yields a
        single claim carrying the hypothesis description, so downstream code
        can always assume a non-empty list.
        """
        known_refs = {
            (c.get("paper_id") or "")[:8] for c in (rag_context or [])
        } | {""}

        claims: list[Claim] = []
        for raw in item.get("claims", []) or []:
            if not isinstance(raw, dict):
                continue
            statement = ensure_str(raw.get("statement", "")).strip()
            if not statement:
                continue

            evidence: list[Evidence] = []
            for ev in raw.get("evidence", []) or []:
                if not isinstance(ev, dict):
                    continue
                try:
                    polarity = int(ev.get("polarity", 1))
                except (TypeError, ValueError):
                    polarity = 1
                polarity = polarity if polarity in (-1, 0, 1) else 1

                try:
                    conf = float(ev.get("confidence", 0.5))
                except (TypeError, ValueError):
                    conf = 0.5
                conf = min(1.0, max(0.0, conf))

                source_ref = ensure_str(ev.get("source_ref", "")).strip("[] ")
                source_type = ensure_str(ev.get("source_type", "prior")).lower()
                if source_type not in ("rag", "citation", "prior", "experiment"):
                    source_type = "prior"
                # A citation the model invented is worse than no citation:
                # downgrade unresolvable refs to "prior" so the provenance
                # chain never claims support it cannot produce.
                if source_type in ("rag", "citation") and source_ref not in known_refs:
                    source_type, source_ref = "prior", ""

                evidence.append(Evidence(
                    text=ensure_str(ev.get("text", ""))[:1000],
                    source_type=source_type,
                    source_ref=source_ref,
                    polarity=polarity,
                    confidence=conf,
                ))

            try:
                claim_conf = float(raw.get("confidence", 0.5))
            except (TypeError, ValueError):
                claim_conf = 0.5

            claims.append(Claim(
                statement=statement,
                evidence=evidence,
                confidence=min(1.0, max(0.0, claim_conf)),
            ))

        if not claims:
            fallback = ensure_str(item.get("description", "")).strip()
            if fallback:
                claims = [Claim(statement=fallback, confidence=0.5)]

        return claims

    async def _generate_simulated(self, goal: ResearchGoal, count: int) -> list[Hypothesis]:
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
            reasoning="Le raisonnement repose sur l'observation que les études récentes montrent une corrélation mais pas de causalité claire.",
            mechanism=f"S'appuyant sur la compréhension actuelle de {goal.domain}, ce mécanisme suggère une intégration interdisciplinaire.",
            generation_method="simulated-literature"
        )
        h.testable_predictions = [
            f"Prédiction 1 : Effet observable dans le contexte de {goal.domain}",
            "Prédiction 2 : Conséquence mesurable en aval",
            "Prédiction 3 : Changement de paramètre quantifiable"
        ]
        h.grounding_evidence = ["Analyse de la littérature existante", "Données expérimentales publiées"]
        h.cited_papers = h.grounding_evidence.copy()
        return h

    async def _generate_from_assumptions(self, goal: ResearchGoal, index: int) -> Hypothesis:
        """Generate from iterative assumptions"""
        h = Hypothesis(
            title=f"Hypothèse {index+1} (Basée sur les suppositions) : Nouvelle cible dans {goal.domain}",
            description=f"Cette hypothèse identifie une nouvelle cible thérapeutique potentielle dans {goal.domain}.",
            reasoning="Si l'on suppose que la cible Z est en fait un effet et non une cause, l'analyse logique pointe vers un régulateur amont.",
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
