# Rapport d’analyse — résultats machine learning (modalité voix)

**Contexte :** modèle tabulaire entraîné sur le jeu Oxford / UCI (`parkinsons.data`), validation **GroupKFold** (5 plis) par **groupe patient** approximé à partir de la colonne `name`. Les métriques ci-dessous proviennent de l’artefact local `models/voice_parkinsons_tabular.joblib` (champ `metrics`).

**Date du rapport :** généré à partir de l’état actuel du dépôt local ; si tu ré-entraînes le modèle, les chiffres peuvent légèrement varier (aléas sklearn / versions).

---

## 1. Données et contexte expérimental

| Indicateur | Valeur | Lecture rapide |
|------------|--------|------------------|
| Nombre d’instances | **195** | Dataset **petit** pour le deep learning ; raisonnable pour des modèles tabulaires classiques avec prudence. |
| Classe 1 (Parkinson, `status=1`) | **147** | Majorité des lignes. |
| Classe 0 (contrôle, `status=0`) | **48** | Minorité (~24,6 % des lignes). |
| Nombre de variables | **22** | Biomarqueurs vocaux déjà extraits (pas d’audio brut). |

**Conséquence pour l’interprétation :** un modèle qui prédit toujours « Parkinson » obtiendrait une **accuracy** artificiellement élevée (~75 %). C’est pourquoi le notebook et la doc insistent sur la **balanced accuracy**, le **rappel** par classe et surtout le **ROC-AUC** en validation croisée **par patient**, plutôt que sur un simple tirage aléatoire de lignes.

---

## 2. Protocole d’évaluation

- **GroupKFold (k = 5)** : à chaque pli, aucun enregistrement d’un même **groupe** (dérivé de `name` en retirant le suffixe numérique) ne se retrouve à la fois en apprentissage et en test.
- **Objectif :** limiter la **fuite d’information** (*data leakage*) entre enregistrements du même sujet, qui gonflerait artificiellement les scores.
- **Métrique principale de comparaison :** **ROC-AUC** moyen sur les 5 plis (capacité à classer les positifs au-dessus des négatifs sur le score continu).
- **Règle de choix du modèle déployé :** maximiser `mean_roc_auc - 0.35 × std_roc_auc` pour favoriser un compromis entre **performance moyenne** et **stabilité** entre plis (un modèle très bon en moyenne mais très variable d’un pli à l’autre est moins fiable pour une démo pédagogique).

---

## 3. Tableau des résultats (classement par score de sélection)

Les valeurs sont celles enregistrées dans `artifact["metrics"]["leaderboard"]`.

| Rang | Modèle | ROC-AUC moyen | Écart-type (entre plis) | Score de sélection `mean − 0,35×std` |
|------|--------|---------------|-------------------------|----------------------------------------|
| 1 | **ExtraTrees** | **0,829** | 0,143 | **0,779** |
| 2 | Random Forest | 0,814 | 0,173 | 0,754 |
| 3 | XGBoost | 0,803 | 0,210 | 0,729 |
| 4 | Gradient Boosting | 0,778 | 0,186 | 0,713 |
| 5 | SVM RBF | 0,736 | 0,134 | 0,690 |
| 6 | Régression logistique | 0,721 | 0,155 | 0,667 |

### Lecture ligne par ligne

- **ExtraTrees (retenu)** : meilleur couple « niveau moyen + relative stabilité » selon la règle choisie. L’écart-type (~0,14) reste **non négligeable** : avec seulement 195 lignes et des groupes patients limités, la variance entre plis est attendue.
- **Random Forest** : très proche d’ExtraTrees ; légèrement moins bon sur le score composite à cause d’une variance un peu plus forte.
- **XGBoost** : ROC-AUC moyen honorable (**0,803**) mais **variance la plus élevée** (~0,21). C’est typique d’un modèle puissant sur un petit jeu : il peut mieux « coller » à certains plis au prix d’instabilité. La règle de pénalisation le classe donc 3ᵉ, ce qui illustre bien le **compromis** « score brut vs robustesse ».
- **Gradient Boosting** (sklearn) : performances intermédiaires, variance modérée.
- **SVM RBF** : AUC moyenne plus basse ; sur petits jeux tabulaires, le noyau RBF peut être sensible au réglage et au scaling (ici géré par pipeline quand présent).
- **Régression logistique** : **baseline** utile ; elle reste en dessous des forêts / boosting sur ce problème non linéaire, ce qui est cohérent avec la nature des descripteurs.

---

## 4. Synthèse de l’analyse

1. **Séparation des classes :** tous les modèles « avancés » (forêts, boosting, XGBoost) dépassent nettement la régression logistique en ROC-AUC moyen, ce qui suggère des **relations non linéaires** et des **interactions** entre biomarqueurs mieux captées par des arbres.
2. **Risque de sur-interprétation :** un ROC-AUC autour de **0,83** en validation par groupe est **encourageant pour un prototype**, mais le jeu est **trop petit et trop spécifique** (Oxford, mesures pré-calculées) pour généraliser à d’autres populations ou à de l’audio en conditions réelles.
3. **Choix ExtraTrees vs XGBoost :** ce n’est pas « XGBoost est moins bon », c’est « **à variance égale ou inférieure**, on préfère ; ici ExtraTrees gagne sur la **stabilité** relative ». Pour un autre tirage ou un autre `random_state`, le classement pourrait bouger légèrement — d’où l’intérêt de **fixer les graines** et de documenter le protocole (déjà fait dans le script d’entraînement).
4. **Limites méthodologiques persistantes :** approximation du groupe patient ; pas de jeu de test **totalement indépendant** dans le temps ; pas de calibration clinique des probabilités ; pas de mesure du bruit d’acquisition.

---

## 5. Augmentation SMOTE (optionnelle)

Le script `scripts/train_voice_tabular.py` accepte `--augment-cv` (SMOTE sur chaque **train** de pli ; le **test** reste réel) et `--augment-final` (fit final sur données suréchantillonnées). Les tableaux des sections 2–3 reflètent une exécution **sans** ces options ; voir `artifact["metrics"]["augmentation"]` après entraînement.

| Mode | Lecture rapide |
|------|----------------|
| Défaut | Métriques et modèle final sur **données réelles** uniquement. |
| `--augment-cv` | Leaderboard potentiellement différent ; expérimentation sur déséquilibre sans toucher au test. |
| `--augment-final` | `joblib` entraîné aussi sur des **exemples synthétiques** ; à signaler explicitement en démo. |

Pour un comparatif chiffré : relancer le script avec et sans `--augment-cv`, puis mettre à jour les tableaux ci-dessus.

---

## 6. Recommandations pratiques

- Pour un exposé ou un rapport de cours : **afficher le tableau complet** (pas seulement le modèle final) et expliquer la règle `mean − 0,35×std`.
- Pour aller plus loin : ajouter **balanced accuracy** et **matrice de confusion** **par pli** dans le notebook ; tester **GroupShuffleSplit** en complément ; tracer des **courbes ROC** par pli (lecture visuelle de la variabilité).
- Pour la production (hors scope hackathon) : collecte plus large, protocole audio, **calibration** des scores, audit juridique / médical.

---

## 7. Références dans le projet

- Entraînement : ce dossier — `scripts/train_voice_tabular.py`, `docs/entrainement_voix.md`, `notebooks/voice/02_train_voice_model_conforme.ipynb`.
- API Flask / contrat JSON : dépôt voisin `multimodal-parkinson-detection-main/docs/voice.md`.
