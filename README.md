# AI Co-Scientist : Système Multi-Agent pour la Découverte Scientifique (v2.0)

Une implémentation du système **AI Co-Scientist**, inspirée par les travaux de **Sakana.ai** ("The AI Scientist") et le papier de **Google DeepMind** "Towards an AI co-scientist" (2025).

> **Mise à jour Février 2026 (v2.0)** : Intégration d'un système **Agentic RAG** complet avec téléchargement de PDFs, indexation vectorielle locale (ChromaDB) et raffinement itératif des hypothèses.

## 🎯 Vue d'ensemble

Ce système est une architecture multi-agent conçue pour :
- **Rechercher** et lire la littérature scientifique (RAG sur ArXiv avec analyse PDF complète).
- **Générer** des hypothèses scientifiques novelles et fondées ("grounded").
- **Évaluer** la qualité, la nouveauté et la testabilité via un "Peer Review" simulé.
- **Débattre** et **classer** les hypothèses via un tournoi (système Elo).
- **Évoluer** les meilleures idées via des stratégies créatives assistées par LLM.
- **Synthétiser** les résultats dans un rapport de méta-revue complet.

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

### 5. **Supervisor & Meta-Agents**
- **Supervisor** : Orchestre le flux de travail asynchrone.
- **Ranking Agent** : Organise des tournois Elo entre hypothèses.
- **Meta-Review Agent** : Rédige le rapport final de la session.

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

*   **Mode "Agentic RAG"** : Le système ne se contente pas de résumés. Il lit le contenu intégral des papiers pour trouver des détails méthodologiques ou des résultats spécifiques ignorés dans les abstracts.
*   **Persistance** : Tous les résultats et l'index vectoriel sont sauvegardés localement. Vous pouvez fermer et relancer l'application sans perdre le contexte.

## 📝 Auteurs & Références

*   Basé sur le framework "AI Co-Scientist" de Google DeepMind (2025).
*   Adapté et étendu avec une couche RAG locale pour une exécution autonome.

**Version** : 2.0 (Février 2026)
**Statut** : Stable
