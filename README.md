# AI Co-Scientist : Système Multi-Agent pour la Découverte Scientifique (v3.1)

Une implémentation du système **AI Co-Scientist**, inspirée par les travaux de **Sakana.ai** ("The AI Scientist") et le papier de **Google DeepMind** "Towards an AI co-scientist" (2025).

> **Mise à jour juillet 2026 (v3.1)** : Fermeture de la boucle empirique
> (adjudication prédiction↔mesure), isolation d'exécution par conteneur,
> classement bayésien Bradley-Terry avec incertitude explicite, orchestration
> par DAG, réplication multivers et intégration Semantic Scholar.
>
> ⚠ **À lire avant de modifier les mécanismes d'évaluation** :
> [`docs/EPISTEMIC-DESIGN.md`](docs/EPISTEMIC-DESIGN.md). Chaque contrainte qui
> y est décrite corrige un mode de défaillance observé ; les retirer les
> réintroduit.

## 🎯 Vue d'ensemble

Ce système est une architecture multi-agent conçue pour :
- **Rechercher** et lire la littérature scientifique (RAG sur ArXiv/PubMed avec analyse PDF).
- **Générer** des hypothèses scientifiques fondées via un modèle de contexte hybride.
- **Évaluer** la qualité via un "Peer Review" simulé, et la nouveauté par recherche d'antériorité (Semantic Scholar).
- **Débattre** et **classer** les hypothèses via un tournoi bayésien (Bradley-Terry avec incertitude).
- **Évoluer** les meilleures idées via des stratégies créatives assistées par LLM.
- **Tester** les prédictions préenregistrées et **adjuger** les réfutations sur les mesures.
- **Synthétiser** les résultats dans un rapport de méta-revue et un **article scientifique PDF**.

## 🏗️ Architecture & Agents

### 1. **Literature Agent (RAG Avancé)**
- **Recherche** : Interroge l'API ArXiv pour trouver les derniers papiers.
- **Lecture** : Télécharge automatiquement les PDFs.
- **Indexation** : Découpe le texte en segments sémantiques et les stocke dans **ChromaDB**.
- **Retrieval** : Fournit aux autres agents des passages précis (preuves) pour étayer chaque affirmation.

### 2. **Generation Agent (Self-Refining)**
- Utilise le contexte RAG pour proposer des hypothèses.
- Boucle de **Self-Refinement** : L'agent critique et améliore sa propre hypothèse avant de la soumettre.

### 3. **Reflection Agent (Critique)**
- Agit comme un reviewer senior. Évalue :
    - **Correctness** : Validité scientifique.
    - **Novelty** : Originalité par rapport à l'état de l'art.
    - **Testability** : Faisabilité expérimentale.

### 4. **Evolution Agent (Créatif)**
- Utilise le LLM pour appliquer des mutations aux meilleures hypothèses :
    - *Simplification* (Rasoir d'Ockham).
    - *Enrichissement* (Ajout de preuves RAG).
    - *Pensée Divergente* (Exploration latérale).

### 5. **Experimentation Agent**
- Génère et exécute du code Python pour tester les prédictions préenregistrées.
- **Expériences typées** (`ExperimentKind`) : une simulation peut réfuter, jamais corroborer.
- **Isolation** : exécution en conteneur (`utils.sandbox_runner`) — pas de réseau, système de fichiers en lecture seule, plafonds mémoire/PID/CPU, capacités abandonnées. `utils.safety` est un filtre *qualité*, pas une frontière de sécurité.
- **Adjudication** : chaque mesure est confrontée à sa prédiction (`utils.adjudication`).

### 6. **Preregistration & Replication Agents**
- **Preregistration** : formalise et scelle les prédictions falsifiables (garde anti-HARKing).
- **Replication** : analyse multivers sur 96 spécifications analytiques défendables.

### 7. **Supervisor & Meta-Agents**
- **Supervisor** : exécute un DAG de tâches validé, parallélise les tâches indépendantes, propage les échecs.
- **Ranking Agent** : tournoi Bradley-Terry bayésien, jugement en double passage contre le biais de position.
- **Meta-Review Agent** : rédige le rapport final de la session.

## 📁 Structure du Projet

```text
.
├── agents/             # Agents spécialisés (Literature, Generation, Experiment, etc.)
├── api/                # Serveur FastAPI
├── data/               # Données persistées (mappings, résultats archivés)
│   └── results/        # Sorties JSON de sessions précédentes
├── docs/               # Documentation (Architecture, Quickstart, papiers PDF)
├── frontend/           # Frontend React + Vite + Tailwind
├── models/             # Modèles de données (Hypothesis, ResearchGoal, Memory)
├── scripts/            # Scripts ad-hoc et utilitaires (generate_paper, extract_*, debug_*)
├── tests/              # Suite de tests unitaires et d'intégration
├── utils/              # Adjudication, sandbox, Bradley-Terry, pipeline,
│                       budget, multivers, IMRaD, hygiène, Semantic Scholar
├── app.py              # Interface utilisateur Streamlit
├── co_scientist.py     # Orchestrateur principal
└── config.py           # Configuration centralisée
```

## 🚀 Installation & Démarrage

### Pré-requis
- Python 3.11+
- **Docker ou Podman** — requis pour exécuter les expériences. Sans runtime de
  conteneur, l'exécution est refusée plutôt que de retomber sur les privilèges
  complets de l'utilisateur. Voir [`docs/OPERATIONS.md`](docs/OPERATIONS.md).
- Un environnement virtuel est recommandé.

### 1. Installation
```powershell
# Cloner le dépôt
git clone https://github.com/NKVD69/NewAIScientist.git
cd NewAIScientist

# Créer un environnement virtuel (si nécessaire)
python -m venv .venv
.venv\Scripts\activate

# Installer les dépendances (incluant ChromaDB, PyPDF, etc.)
pip install -r requirements.txt
```

### 2. Configuration LLM (Local ou API)
Le système est pré-configuré pour fonctionner avec **LM Studio** ou **Ollama** en local.
- **URL par défaut** : `http://127.0.0.1:1234/v1`
- **Modèle** : Configurable dans l'interface (ex: `mistral-7b`, `llama-3`).

### 3. Lancement de l'Interface
Utilisez la commande suivante pour lancer l'application Streamlit :

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

## 🖥️ Utilisation de l'Interface

1.  **Sidebar** :
    *   **Utiliser LLM Local** : ✅ Activé.
    *   **Activer RAG** : ✅ Cochez pour activer l'analyse profonde des PDFs.
2.  **Objectif de Recherche** :
    *   Cliquez sur **"🪄 Auto-détecter"** pour remplir les champs à partir d'une simple phrase.
    *   Exemple : *"Trouver de nouvelles cibles thérapeutiques pour le glioblastome."*
3.  **Lancer** :
    *   Suivez la progression dans les logs (Recherche -> Lecture PDF -> Génération -> Tournoi).
4.  **Résultats** :
    *   Explorez les onglets **Hypothèses**, **Littérature** (sources PDF), et **Meta-Review**.

## 🧠 Fonctionnalités Avancées

*   **Modèle "Hybrid Context"** : Le système combine trois couches de connaissances :
    *   **CAG (Context-Augmented Generation)** : Synthèse des découvertes clés de la littérature.
    *   **GraphRAG** : Extraction d'entités et de relations pour un raisonnement structurel.
    *   **Agentic RAG** : Recherche sémantique profonde dans le texte intégral des PDFs.
*   **Génération d'Articles PDF** : Un script dédié (`generate_paper.py`) permet de transformer les résultats d'une session en un article scientifique structuré (Abstract, Architecture, Study Case, Future Directions).
*   **Persistance** : Tous les résultats et l'index vectoriel sont sauvegardés localement.

## 📚 Documentation

| Document | Contenu |
|---|---|
| [`docs/EPISTEMIC-DESIGN.md`](docs/EPISTEMIC-DESIGN.md) | **À lire avant de modifier l'évaluation, le classement ou l'expérimentation.** Le raisonnement derrière chaque contrainte. |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | Variables d'environnement, sandbox, budget, arbitrages de coût, lecture d'un run |
| [`docs/SEMANTIC-SCHOLAR.md`](docs/SEMANTIC-SCHOLAR.md) | Intégration S2, nouveauté ancrée, pièges |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Architecture générale |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Prise en main |
| [`docs/AUDIT-2026-07.md`](docs/AUDIT-2026-07.md) | Instantané historique de l'état pré-correctifs |

## 🚧 Limites connues

- **Persistance** : l'état vit en mémoire et en JSON. `api/server.py` détient
  une instance globale unique — ni multi-session, ni multi-utilisateur, perte
  d'état au redémarrage.
- **Reproductibilité des appels LLM** : aucune graine, aucun enregistrement de
  `(model, temperature, prompt_hash)`. Deux exécutions du même objectif restent
  incomparables.
- **Triple surface d'interface** : Streamlit, FastAPI et React réimplémentent
  chacune l'orchestration des phases.
- **Semantic Scholar** : le client suit le schéma publié mais n'a pas été
  exercé contre l'API live. Voir la section « État de test » de
  [`docs/SEMANTIC-SCHOLAR.md`](docs/SEMANTIC-SCHOLAR.md).

## 📝 Auteurs & Références

*   Basé sur le framework "AI Co-Scientist" de Google DeepMind (2025).
*   Adapté et étendu avec une couche RAG locale pour une exécution autonome.

**Version** : 3.1 (juillet 2026)
**Statut** : Stable & Modulaire · 394 tests
