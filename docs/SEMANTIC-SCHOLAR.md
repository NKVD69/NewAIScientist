# Intégration Semantic Scholar

`utils/semantic_scholar.py` — client de la Graph API.
`utils/novelty.py` — évaluation de nouveauté ancrée sur une recherche d'antériorité.

## Ce que S2 apporte que les autres sources ne peuvent pas

### Comptes de citations

`literature_hygiene.quality_weight` lit un champ `citation_count`. Ni l'API
arXiv ni les E-utilities PubMed ne le renvoient : la branche était morte.

Un appel batché (jusqu'à 500 articles) le remplit pour tout le corpus. Le
client préfère `influentialCitationCount` à `citationCount` — il exclut les
citations de complaisance et constitue un meilleur proxy de qualité.

### Types de publication

S2 étiquette `MetaAnalysis`, `ClinicalTrial`, `Review`, `CaseReport`… C'est une
hiérarchie de preuve, et les traiter comme équivalents est une erreur de fond
en contexte biomédical.

```
quality_weight (MetaAnalysis, 2024, 38 citations influentes) : 1.63
quality_weight (CaseReport,   2024,  2 citations)            : 0.675
```

Un facteur 2,4 qui n'existait pas : tout pesait 1,0. Barème dans
`PUBLICATION_TYPE_WEIGHT`.

### Recherche sémantique

C'est ce qui rend possible une vraie vérification d'antériorité. Voir
[EPISTEMIC-DESIGN.md](EPISTEMIC-DESIGN.md#6-la-nouveauté-se-cherche-elle-ne-sintrospecte-pas).

## Configuration

```bash
export S2_API_KEY="votre-clé"   # https://www.semanticscholar.org/product/api
```

Sans clé, le trafic partage un petit pool (~100 req / 5 min) et prend des 429
facilement ; le client s'auto-limite à 0,34 req/s. Avec clé : 1 req/s, backoff
exponentiel sur 429. Le rate limiter est partagé via `get_client()` — ne pas
instancier `SemanticScholarClient()` en boucle, cela contourne la limitation.

## Utilisation

### Comme source de littérature

```python
await co.literature_agent.search_literature(
    goal, sources=["arxiv", "pubmed", "semanticscholar"],
)
```

Contrairement à l'adaptateur PubMed, S2 ne force **pas** de filtre
open-access. Ce filtre biaise le corpus vers certains éditeurs ; c'est un choix
qui doit rester réversible (`open_access_only=True`), pas une contrainte
invisible dans chaque requête.

### Enrichissement d'un corpus existant

```python
from utils.semantic_scholar import get_client
papers = await get_client().enrich_papers(papers)   # un seul appel batché
```

Résout chaque article par DOI, arXiv ID ou PMID. Les records non résolus
passent inchangés — cette étape ne doit jamais supprimer d'article.

Activé par défaut dans `LiteratureAgent` ; désactivable via
`LiteratureAgent(enrich_with_s2=False)`.

### API directe

```python
from utils.semantic_scholar import SemanticScholarClient

client = SemanticScholarClient()

# Recherche par pertinence, avec filtres
papers = await client.search("KRAS G12C resistance", limit=20,
                             year="2020-", min_citation_count=5,
                             publication_types=["ClinicalTrial"])

# Recherche bulk : syntaxe booléenne, pagination par token, jusqu'à 1000
papers = await client.search_bulk("KRAS + (inhibitor | degrader)", limit=500)

# Lookup — préfixes acceptés : DOI: ARXIV: PMID: PMCID: CorpusId:
paper  = await client.get_paper("DOI:10.1038/s41586-024-0001")
papers = await client.get_papers_batch([...])        # 500 max par appel

# Traversée du graphe
refs  = await client.get_references("DOI:10.1038/...")
cites = await client.get_citations("DOI:10.1038/...")   # inclut les réfutations
recs  = await client.recommend("DOI:10.1038/...")
```

### Nouveauté

```python
from utils.novelty import assess_novelty, apply_report

report = await assess_novelty(hypothesis, rag_engine=rag)
apply_report(hypothesis, report)
print(report.render())
```

Appelé automatiquement en début de `run_review_cycle`, en batch avec
concurrence bornée.

## Pièges

**Le paramètre `fields`.** S2 ne renvoie que `paperId` s'il n'est pas fourni.
Si des champs remontent vides, c'est la première chose à regarder. Voir
`DEFAULT_FIELDS`.

**Les nulls explicites.** S2 renvoie `"tldr": null` plutôt que d'omettre la
clé. `S2Paper.from_api` le gère ; du code appelant qui fait
`raw["tldr"]["text"]` ne le gère pas.

**Les DOI.** Le code registrant fait 4 à 9 chiffres. `10.1/x` n'est pas un DOI
valide — une regex qui l'accepte produira des faux positifs.

**Concurrence.** `assess_many()` borne à 3 requêtes simultanées. Augmenter cette
valeur sans clé API garantit des 429.

## ⚠ État de test

L'API live n'était pas joignable depuis l'environnement de développement
(`api.semanticscholar.org` hors allowlist réseau). Les 40 tests de
`tests/test_semantic_scholar.py` mockent chaque appel à la frontière
`_request`.

**Couvert** : parsing, résolution d'identifiants, batching, pagination,
seuils, rate limiting, dégradation en cas d'échec.

**Non couvert** : le contrat réel de l'API. Les formes d'endpoints et les noms
de champs suivent le schéma publié, mais le comportement sous 429 réel, les
champs `null` inattendus et les limites de pagination bulk n'ont pas été
exercés.

Vérification manuelle recommandée :

```python
import asyncio
from utils.semantic_scholar import SemanticScholarClient

async def check():
    c = SemanticScholarClient()
    for p in await c.search("CRISPR base editing", limit=5):
        print(f"{p.year} {p.citation_count:>5} cit · {p.publication_types} · {p.title[:60]}")
    print(c.stats())

asyncio.run(check())
```
