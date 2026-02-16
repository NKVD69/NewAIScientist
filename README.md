# AI Co-Scientist: Multi-Agent System for Scientific Discovery

Une implémentation du système **AI co-scientist**, inspirée par les travaux de **Sakana.ai** ("The AI Scientist") et le papier de **Google DeepMind** "Towards an AI co-scientist" (2025).

## 🎯 Vue d'ensemble

Ce système est une architecture multi-agent conçue pour :
- **Rechercher** et analyser la littérature scientifique existante (RAG avec ArXiv).
- **Générer** des hypothèses scientifiques novelles et fondées ("grounded").
- **Évaluer** la qualité, la nouveauté et la testabilité.
- **Débattre** et **classer** les hypothèses via un tournoi (système Elo).
- **Évoluer** et **améliorer** les hypothèses itérativement.
- **Synthétiser** les insights et fournir une vue d'ensemble de la recherche.

## 🏗️ Architecture

### Agents Spécialisés

#### 1. **Literature Agent (Nouveau)**
- Interroge l'API **ArXiv** pour trouver des papiers pertinents en Open Access.
- Analyse les résumés pour fournir un contexte scientifique réel au système.

#### 2. **Generation Agent** 
- **Mode RAG** : Utilise le contexte bibliographique fourni par le Literature Agent pour générer des hypothèses ancrées dans la réalité.
- **Mode LLM** : Appelle un LLM local (Ollama, LM Studio) pour la créativité.

#### 3. **Reflection Agent**
- Agit comme un "reviewer scientifique senior". Analyse l'hypothèse et retourne une critique détaillée ainsi que des scores précis (Correctness, Novelty, Testability, Quality).

#### 4. **Ranking Agent**
- Classe les hypothèses via un **tournoi Elo** en simulant des débats scientifiques.

#### 5. **Proximity Agent**
- Calcule la **similarité** entre les hypothèses pour le clustering et la déduplication.

#### 6. **Evolution Agent**
- Améliore les hypothèses via des stratégies comme l'enrichissement, la simplification ou la combinaison.

#### 7. **Meta-Review Agent**
- Synthétise les résultats, identifie les tendances et génère un aperçu de la recherche.

#### 8. **Supervisor Agent**
- Orchestre tous les agents et gère une file de tâches asynchrone.

## 🚀 Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/your-repo/ai-co-scientist.git
cd ai-co-scientist

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. (Optionnel) Configurer un LLM local (voir section ci-dessous)
```

## 🖥️ Interface Graphique (GUI)

Une interface moderne basée sur **Streamlit** est disponible pour piloter l'assistant sans toucher au code.

### Lancer l'interface

```bash
streamlit run app.py
```

### Fonctionnalités
- **Configuration** : Activez/Désactivez le LLM et configurez l'URL (Ollama/LM Studio) directement depuis la barre latérale.
- **Tableau de Bord** : Suivez la génération, la revue et les tournois en temps réel.
- **Littérature** : Visualisez les papiers ArXiv récupérés et utilisés pour la génération.
- **Visualisation** : Graphiques interactifs des scores Elo et de la distribution Qualité/Nouveauté.
- **Exploration** : Inspectez chaque hypothèse, ses critiques et ses preuves.
- **Export** : Téléchargez le rapport complet en JSON.

## 🧠 Connexion à un LLM Local (Ollama/LM Studio)

Le système est conçu pour fonctionner avec un LLM local via une API compatible OpenAI. Cela alimente à la fois la **génération** et la **critique** (review) des hypothèses.

1.  **Démarrez votre serveur LLM** :
    *   **LM Studio** : Allez dans l'onglet "Local Server" et démarrez le serveur (port 1234 par défaut).
    *   **Ollama** : Lancez `ollama serve` (port 11434 par défaut).

2.  **Configuration** :
    *   **Via l'interface (recommandé)** : Entrez simplement l'URL et le nom du modèle dans la barre latérale de l'application Streamlit.
        *   URL par défaut : `http://127.0.0.1:1234/v1`
        *   Modèle par défaut : `openai/gpt-oss-20b`
    *   **Via CLI** : Configurez les variables d'environnement :
        ```bash
        export OPENAI_API_BASE="http://127.0.0.1:1234/v1"
        export OPENAI_MODEL_NAME="openai/gpt-oss-20b"
        ```

Si aucun LLM n'est détecté, le système basculera automatiquement en mode simulé pour chaque agent.

## 💻 Utilisation en Ligne de Commande (CLI)

Si vous préférez utiliser le script sans interface graphique :

```bash
python co_scientist.py
```

Ce script exécute un cycle de recherche complet sur un cas d'usage prédéfini (repositionnement de médicaments pour la leucémie) et exporte les résultats dans `co_scientist_results.json`.

### Sortie Attendue

Lorsque le LLM local est connecté, vous verrez :
```
✓ Generation Agent initialized with local LLM connection.
✓ Reflection Agent initialized with local LLM connection.

📚 Running literature search...
✓ Found 5 relevant papers.

🔬 Generating 5 initial hypotheses...
...
```

## 🔧 Personnalisation

### Changer le Modèle LLM

Le modèle utilisé est défini dans `GenerationAgent` et `ReflectionAgent`. Par défaut, il est réglé sur `"openai/gpt-oss-20b"`. Vous pouvez le changer pour tout autre modèle que vous servez localement.

```python
# In co_scientist.py -> GenerationAgent -> _generate_with_llm
response = await asyncio.to_thread(
    self.llm_client.chat.completions.create,
    model="llama3",  # Change this to your model
    ...
)
```

### Désactiver le LLM

Pour forcer le mode de simulation, initialisez `CoScientist` avec `use_local_llm=False`.

```python
# In co_scientist.py -> main()
co_scientist = CoScientist(use_local_llm=False)
```

## 📝 Limites et Considérations

- **Qualité LLM** : La pertinence des hypothèses et des critiques dépend fortement du modèle utilisé.
- **Ranking Agent** : L'agent de classement (`RankingAgent`) fonctionne encore en mode simulé (calcul de scores heuristiques). L'intégration du LLM pour simuler les débats est la prochaine étape.
- **Accès aux Données** : Le système utilise l'API ArXiv publique. Assurez-vous d'avoir une connexion internet active.

## 🎓 Références

- **Paper** : "Towards an AI co-scientist" - Google DeepMind (2025)
- **Authors** : Gottweis et al.

---

**Status** : ✅ Fonctionnel (Hybride LLM/Simulation + RAG ArXiv + GUI)
**Dernière mise à jour** : Janvier 2026
**Auteur** : Reproduction du framework co-scientist
