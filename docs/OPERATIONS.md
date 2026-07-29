# Exploitation

Variables d'environnement, prérequis et arbitrages de coût.

## Sandbox d'exécution

**Prérequis : Docker ou Podman.** Sans runtime, les expériences ne s'exécutent
pas — le système refuse plutôt que de retomber sur les privilèges complets de
l'utilisateur. Voir
[EPISTEMIC-DESIGN.md](EPISTEMIC-DESIGN.md#3-lisolation-vient-du-noyau-pas-dune-liste-dimports).

```bash
docker pull python:3.12-slim
python -c "from utils.sandbox_runner import isolation_report; print(isolation_report())"
```

| Variable | Défaut | Effet |
|---|---|---|
| `NEWAISCI_SANDBOX_IMAGE` | `python:3.12-slim` | Image du conteneur |
| `NEWAISCI_SANDBOX_TIMEOUT` | `30` | Budget d'exécution (s) |
| `NEWAISCI_SANDBOX_MEMORY_MB` | `512` | Plafond mémoire (swap aligné) |
| `NEWAISCI_SANDBOX_PIDS` | `64` | Plafond de processus |
| `NEWAISCI_SANDBOX_CPUS` | `1.0` | Part CPU |
| `NEWAISCI_SANDBOX_NETWORK` | `0` | `1` active le réseau — à éviter |
| `NEWAISCI_ALLOW_UNSANDBOXED` | `0` | `1` accepte le repli `rlimit`, nettement plus faible |

Une image contenant numpy/scipy/pandas/sklearn évite que chaque run échoue sur
un `ImportError` :

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy scipy pandas scikit-learn
```

## Budget LLM

Désactivé par défaut : sans variable, aucune limite. C'est délibéré — activer
un plafond par défaut casserait des runs existants. En production,
`NEWAISCI_MAX_COST_USD` est le garde-fou minimal.

| Variable | Effet |
|---|---|
| `NEWAISCI_MAX_TOKENS` | Plafond de tokens cumulés |
| `NEWAISCI_MAX_LLM_CALLS` | Plafond d'appels |
| `NEWAISCI_MAX_COST_USD` | Plafond de coût |
| `NEWAISCI_MAX_PROMPT_TOKENS` | Plafond par appel — détecte l'accumulation de contexte |

Le disjoncteur lève `BudgetExhausted` avant l'appel plutôt que de renvoyer
`None` : un appel silencieusement sauté produit un artefact à moitié construit
qui a l'air complet. Le pipeline vérifie le budget entre chaque vague.

`BudgetTracker.render()` attribue la dépense par rôle.

## Semantic Scholar

| Variable | Effet |
|---|---|
| `S2_API_KEY` / `SEMANTIC_SCHOLAR_API_KEY` | Clé API — 1 req/s au lieu de 0,34 |

## Embeddings

| Variable | Effet |
|---|---|
| `NEWAISCI_EMBEDDING_MODEL` | Force un modèle précis |

Cascade par défaut : BioLORD-2023 → S-PubMedBert-MS-MARCO → SPECTER2 →
all-MiniLM-L6-v2. Les modèles de domaine pèsent plusieurs centaines de Mo au
premier chargement ; le repli généraliste émet un avertissement explicite.

## Arbitrages de coût

**Tournoi.** Le double passage du juge double les appels, et
`recommended_budget` dimensionne à `2·n·log₂n`. Pour 14 hypothèses : 107 matchs
× 2 passages, contre 12 auparavant. C'est le prix d'un classement identifié
plutôt que du bruit. Trois leviers :

```python
RankingAgent(two_sided_judging=False)          # perd la correction du biais
co.run_tournament_cycle(num_matches=20)        # budget explicite
co.run_tournament_cycle(stop_when_separated=True)   # défaut, coupe souvent tôt
```

**Multivers.** 32 spécifications par hypothèse au lieu de 3 réplications. Le
coût est en exécutions sandbox, pas en appels LLM — le code est généré une fois
et varié par le harnais. Régler via
`ReplicationAgent(max_specifications=16)`.

**Screening de rétractations.** Une requête Crossref ou PubMed par article,
concurrence 4. Désactivable via `LiteratureAgent(screen_retractions=False)`. Un
échec de lookup dégrade gracieusement : l'article est conservé, pas exclu.

**Simulations.** `ExperimentAgent(allow_simulation=False)` transforme « aucune
donnée pertinente » en `INFEASIBLE` explicite. Recommandé en production ;
laissé à `True` par défaut pour ne pas casser les runs existants.

## Lecture d'un run

Chaque run se termine par un bloc RUN HEALTH. La ligne qui compte :

```
⚠ INCOMPLETE RUN — some tasks failed or were skipped. Any manuscript or
  protocol produced above rests on a partial evidence base and must state so.
```

Un run non `clean` a produit des artefacts sur une base incomplète. Le rapport
de pipeline indique quelle tâche a échoué et lesquelles ont été sautées en
conséquence.

Vérifier aussi le taux d'invariance du juge :

```
✓ 24 matches. Judge order-invariance: 47%
  ⚠ The judge changes its mind when A and B are swapped more often than not.
```

Sous 60 %, le classement ne doit pas être cru.
