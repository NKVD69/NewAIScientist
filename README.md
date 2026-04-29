# AI Co-Scientist : Système Multi-Agent pour la Découverte Scientifique (v2.2)

Une implémentation du système **AI Co-Scientist**, inspirée par les travaux de **Sakana.ai** ("The AI Scientist") et le papier de **Google DeepMind** "Towards an AI co-scientist" (2025).

> **Mise à jour Avril 2026 (v2.2)** : Refonte modulaire de l'architecture, introduction de l'agent d'**Expérimentation** avec contrôle de sécurité AST, et optimisation du modèle **Hybrid Context**.

## 🎯 Vue d'ensemble

Ce système est une architecture multi-agent conçue pour :
- **Rechercher** et lire la littérature scientifique (RAG sur ArXiv/PubMed avec analyse PDF).
- **Générer** des hypothèses scientifiques fondées via un modèle de contexte hybride.
- **Évaluer** la qualité, la nouveauté et la testabilité via un "Peer Review" simulé.
- **Débattre** et **classer** les hypothèses via un tournoi (système Elo).
- **Évoluer** les meilleures idées via des stratégies créatives assistées par LLM.
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

### 5. **Experimentation Agent (Nouveau)**
- Génère et exécute du code Python pour tester les prédictions des hypothèses.
- **Sécurité** : Inclut une couche de vérification AST (`utils.safety`) pour bloquer les opérations système dangereuses avant l'exécution.

### 6. **Supervisor & Meta-Agents**
- **Supervisor** : Orchestre le flux de travail asynchrone et gère la file d'attente des tâches.
- **Ranking Agent** : Organise des tournois Elo entre hypothèses.
- **Meta-Review Agent** : Rédige le rapport final de la session.

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
├── utils/              # Utilitaires (RAG, Sécurité AST, LLM helpers)
├── app.py              # Interface utilisateur Streamlit
├── co_scientist.py     # Orchestrateur principal
└── config.py           # Configuration centralisée
```

## 🚀 Installation & Démarrage

### Pré-requis
- Python 3.9+
- Un environnement virtuel est recommandé.

### 1. Installation
```powershell
# Cloner le dépôt
git clone https://github.com/your-repo/ai-co-scientist.git
cd ai-co-scientist

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

## 📝 Auteurs & Références

*   Basé sur le framework "AI Co-Scientist" de Google DeepMind (2025).
*   Adapté et étendu avec une couche RAG locale pour une exécution autonome.

**Version** : 2.2 (Avril 2026)
**Statut** : Stable & Modulaire
