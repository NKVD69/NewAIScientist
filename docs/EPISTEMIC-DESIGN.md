# Conception épistémique

Ce document explique **pourquoi** certains mécanismes du système sont écrits
comme ils le sont. Il s'adresse à quiconque envisage de les modifier : chacune
des contraintes décrites ici corrige un mode de défaillance observé, et les
retirer les réintroduit.

Le fil conducteur : un co-scientifique qui ne peut pas conclure « non » n'est
pas un co-scientifique, c'est un générateur de récits plausibles — et le reste
de la machinerie ne fait qu'en augmenter la crédibilité apparente.

---

## 1. La réfutation se décide sur des mesures, pas sur du texte

### La règle

`utils/adjudication.py` confronte chaque `Prediction` préenregistrée à sa
mesure observée via `Prediction.is_refuted_by()`. Un verdict par prédiction,
jamais un verdict global sur du texte libre.

### Pourquoi

La version précédente cherchait des sous-chaînes dans le stdout de
l'expérience :

```python
if "fail" in results.lower() or "reject" in results.lower():
    refuted = True
```

Deux défaillances symétriques :

- **Faux positif.** `"failed to reject the null hypothesis"` — la formule
  statistique standard signifiant *absence de preuve contre* — déclenche le
  détecteur deux fois. L'hypothèse était marquée réfutée sur la base d'une
  phrase signifiant l'inverse.
- **Faux négatif.** `"observed IC50 = 47 µM vs predicted 2 µM"` — une
  réfutation quantitative sans ambiguïté — ne contient aucun mot-clé et
  passait silencieusement.

### Ce qu'il ne faut pas faire

- Ne pas rétablir de décision fondée sur le texte de `experimental_results`.
  Ce champ existe pour l'affichage et le `WritingAgent` ; la logique de
  décision lit `hypothesis.verdicts`.
- Ne pas traiter `UNTESTED` comme un soutien. Le silence n'est pas une preuve.
  `ExperimentRun.evidential_weight` renvoie exactement `0.0` quand rien n'a été
  mesuré, et c'est intentionnel.
- Ne pas élargir un `refuting_threshold` après coup pour accommoder une mesure
  décevante. C'est de la rationalisation post-hoc, et le garde anti-HARKing la
  détecte.

### Le contrat

Le script d'expérience doit terminer par une ligne :

```
RESULTS_JSON: {"measurements": [{"quantity": "IC50", "observed": 47.3,
               "unit": "uM", "n": 96, "test": "welch_t", "p_value": 0.9}]}
```

Les noms de quantités doivent reprendre ceux du préenregistrement — c'est ce
qui rend l'appariement fiable. Sans cela le modèle invente ses propres noms et
tout atterrit en `UNTESTED`.

---

## 2. Une simulation peut réfuter, jamais corroborer

### La règle

`ExperimentKind.DRY_RUN_SIMULATION.can_corroborate` vaut `False`. Une mesure
concordante issue d'une simulation produit `CONSISTENT_UNSCORED` et un poids
probant nul.

### Pourquoi

Le chemin par défaut demandait au LLM de générer des données synthétiques
« réalistes », d'y appliquer des t-tests, puis de conclure si les données
soutenaient l'hypothèse. Les p-valeurs obtenues mesurent la cohérence interne
du générateur avec lui-même : zéro bit d'information externe, présenté avec
l'appareil rhétorique complet d'un résultat (tailles d'effet, intervalles de
confiance).

C'est le mode de défaillance le plus dangereux du système, parce que la sortie
est **indiscernable** d'un vrai résultat en aval : le `WritingAgent` la rédige
en Results.

La réfutation reste permise parce qu'une incohérence interne est informative :
si la simulation contredit l'hypothèse qui l'a engendrée, quelque chose ne
tient pas.

### Le garde de pertinence

Des données réelles mais hors-sujet ne valent pas mieux que du synthétique.
Un run PubChem est **rétrogradé en simulation** si les descripteurs
physicochimiques (MW, XLogP, TPSA) ne concernent pas les quantités
enregistrées. Corréler le TPSA avec une hypothèse d'inhibition de prolifération
teste les règles de Lipinski, pas l'hypothèse.

### Hiérarchie des sources

| Source | Type | Peut corroborer |
|---|---|---|
| ChEMBL (IC50/Ki/EC50) | issues d'essai | oui |
| DepMap, LINCS, Open Targets | mesures | oui (non implémenté) |
| PubChem (descripteurs) | pertinent seulement si les prédictions sont physicochimiques | conditionnel |
| Synthétique | — | non |

---

## 3. L'isolation vient du noyau, pas d'une liste d'imports

### La règle

`utils/sandbox_runner.py` est la frontière de sécurité. `utils/safety.py` est
un **filtre qualité** et son docstring le dit explicitement.

### Pourquoi

`check_code_safety` n'inspecte que les nœuds `Import` / `ImportFrom` contre une
liste noire. Dix contournements ont été vérifiés comme passant :

```
importlib.import_module("os")      __import__("os").system("id")
import http.client                 open("/etc/passwd", "w")
exec(compile("import os", ...))    ().__class__.__mro__[1].__subclasses__()
import asyncio / pickle            while True: pass       bytearray(10**10)
```

Une liste blanche statique d'imports est structurellement le mauvais moyen de
défense contre un générateur de code : Python offre une infinité de façons
d'atteindre une capacité sans la nommer dans un `import`.

`tests/test_ranking_and_sandbox.py::TestASTFilterIsNotSecurity` paramètre ces
contournements et **assert qu'ils passent le filtre**, pour que la suite
documente elle-même pourquoi le sandbox existe.

### Fail-closed

Sans runtime de conteneur, l'exécution est **refusée**. Le repli `rlimit`
demande un opt-in explicite (`NEWAISCI_ALLOW_UNSANDBOXED=1`). Ne pas
transformer cela en dégradation silencieuse : l'ancien code exécutait le script
avec l'UID complet de l'utilisateur dès que le filtre passait.

---

## 4. Le classement porte son incertitude

### La règle

`utils/bradley_terry.py` modélise des croyances gaussiennes (μ, σ). Toute
sélection passe par `CoScientist.top_hypotheses()`, qui classe sur **μ − 2σ**.

### Pourquoi

Trois défauts d'Elo se composaient :

**Biais de position non corrigé.** `hyp_a` était toujours présenté comme « A »,
et les égalités tranchées vers A. Le plan de paires n'étant pas mélangé, le
biais était systématique et non moyenné. Correction : chaque paire est jugée
dans les deux ordres, et seuls les critères invariants par permutation
comptent. Un juge purement positionnel produit donc un nul — il ne nous a rien
appris sur la paire, seulement sur lui-même.

**Sous-échantillonnage.** 14 hypothèses, 12 matchs — 1,7 partie chacune, là où
il en faut ~53 pour que le classement soit identifié. Le top-1 qui décidait du
manuscrit était pour l'essentiel tiré au sort. `recommended_budget()`
dimensionne à `2·n·log₂n`.

**Héritage anti-darwinien.** Les hypothèses évoluées naissaient à 1200 alors
que leurs parents — sélectionnés *parce que* les mieux classés — étaient
au-dessus. Avec 1,7 partie, elles ne remontaient jamais. La sélection jouait
contre l'évolution. `inherit()` transmet μ avec régression vers la moyenne et
σ élargi.

### Diagnostic

`RankingAgent.judge_reliability()` expose le taux d'invariance par permutation.
Sous 60 %, le juge lit la position plutôt que le contenu et le classement ne
doit pas être cru. L'ancien code ne pouvait pas le détecter.

---

## 5. Le préenregistrement scelle réellement

`_compute_prediction_hash` ne hache que le contenu des prédictions, triées par
quantité. Le timestamp vit dans `registered_at`, **hors du bundle haché** —
l'y inclure faisait échouer `verify_integrity()` en toute circonstance, rendant
la garantie anti-HARKing vacue.

`check_integrity()` distingue trois cas, parce qu'ils ont des implications
opposées :

```
(True,  'intact (sha256:fadb4d84ddff)')
(False, 'TAMPERED: predictions changed since registration ...')
(False, 'predictions were never registered')
```

`run_revision_cycle` vérifie l'intégrité **avant** d'accorder le moindre crédit
aux verdicts.

---

## 6. La nouveauté se cherche, elle ne s'introspecte pas

### La règle

`utils/novelty.py` lance une recherche d'antériorité Semantic Scholar et
retourne les articles les plus proches, avec titres et liens.

### Pourquoi

Le chemin de repli calculait :

```python
if "llm" in generation_method:         score = 0.75
elif "simulated" in generation_method: score = 0.55
```

La nouveauté était fonction du **mode de génération**. Une hypothèse était
jugée 36 % plus novatrice pour être sortie d'un LLM plutôt que du stub de
simulation — artefact de plomberie promu au rang de mesure, portant un poids de
0,25 dans le prior de classement.

Le chemin LLM demandait au méta-relecteur un `novelty_score` par introspection,
sans aucune récupération, pour précisément le domaine où la mémoire
paramétrique est la plus faible : la littérature récente et de niche.

### Auditable par construction

Un scalaire de 0,75 ne vaut rien pour un relecteur. Le rapport expose les trois
articles les plus proches pour qu'un humain juge lui-même. C'est le but.

### « Inconnu » n'est pas « moyen »

Une recherche qui échoue produit `level="unknown"` et `searched=False`, pas une
valeur par défaut plausible. C'est la distinction que l'ancien code ne savait
pas exprimer, et c'est ainsi qu'un artefact a pu passer pour une mesure.

### Seuils selon l'échelle

Les seuils de signalement dépendent de la méthode de similarité employée :
cosinus d'embeddings et Jaccard de tokens vivent sur des échelles différentes.
Un seuil unique de 0,85 (calibré cosinus) signifiait qu'avec le repli Jaccard
l'antériorité n'était **jamais** signalée. Voir
`NoveltyReport.PRIOR_ART_THRESHOLD`.

---

## 7. La reproductibilité se teste par variation analytique

### La règle

`utils/multiverse.py` énumère 96 spécifications défendables (politique
d'aberrants × test statistique × transformation × ajustement) et rapporte la
distribution des effets.

### Pourquoi

La réplication par variation de seed ne mesurait rien :

- Sur données réelles, l'analyse est déterministe. Variance nulle **par
  construction** : le système rapportait une reproductibilité parfaite, ce qui
  était vrai et sans intérêt.
- Sur données synthétiques, elle mesurait la variance du RNG que le LLM venait
  d'écrire. « Robuste » signifiait qu'il avait choisi un petit écart-type.

La vraie question n'est pas *est-ce que ça se rejoue à l'identique ?* mais
*est-ce que la conclusion survit aux choix analytiques que l'analyste aurait pu
faire autrement ?*

### Attribution

`fork_influence()` répond à la question pour laquelle la specification curve
existe : *quel choix arbitraire pilote le résultat ?* Un fork responsable de
l'essentiel de la dispersion est un fait sur l'analyse, pas sur la nature.

`multiverse_fragility` alimente le prior de classement — un résultat qui ne
survit qu'à 6 spécifications sur 32 est pénalisé, pas rédigé.

---

## 8. Un échec de tâche est visible

### La règle

`utils/pipeline.py` déclare le graphe de dépendances, le valide et propage les
échecs. `PipelineReport.clean` est `False` dès qu'une tâche a échoué ou été
sautée.

### Pourquoi

L'ancien exécuteur stockait les exceptions **comme résultat de tâche** :

```python
except Exception as e:
    task.result = e          # l'exception devient le résultat
```

Un `run_literature_search` mort laissait la génération s'exécuter — sans
littérature, sans CAG, sans index RAG — produisant des hypothèses fluides et
hallucinées qui traversaient tournoi, expérience et rédaction sans qu'aucune
étape ne signale la base documentaire vide.

### Ne pas contourner

- Ne pas passer `on_failure=IGNORE` pour faire taire une tâche instable :
  corriger la tâche, ou utiliser `RETRY`.
- Ne pas retirer le bloc RUN HEALTH. Un run partiel doit être visiblement
  partiel ; c'est la seule chose qui empêche un manuscrit fondé sur rien de
  ressembler à un manuscrit fondé sur quelque chose.

---

## 9. La provenance des preuves compte

`utils/imrad.py` étiquette chaque chunk par section et lui attribue un poids
probant.

Sans cela, une phrase de **Discussion** — où les auteurs spéculent au
conditionnel sur ce que leurs résultats *pourraient* impliquer — était
indiscernable en aval d'une phrase de **Results**, où ils rapportent ce qu'ils
ont mesuré. Comme `GenerationAgent` ancre ses hypothèses dans les chunks
récupérés, la spéculation d'auteur entrait comme fait établi et ressortait en
« grounding evidence ».

Complété par une détection de hedging : une section « Results » écrite
entièrement au conditionnel reste de la spéculation.

`grounding_profile()` avertit quand le soutien d'une hypothèse provient
majoritairement d'Introduction et Discussion.

---

## Principe transversal

Plusieurs de ces mécanismes partagent une même forme : le système doit pouvoir
dire **« non »**, **« je ne peux pas tester ceci »** et **« ceci est une
simulation et ne corrobore rien »**.

Un système qui ne peut produire que des sorties affirmatives paraîtra toujours
productif. C'est précisément ce qui le rend dangereux, et c'est la propriété
que ces contraintes préservent.
