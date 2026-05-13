# Modalité voix — documentation

## 1. Objectif de la modalité voix

Proposer une **évaluation exploratoire** à partir de **biomarqueurs vocaux tabulaires** (nombres déjà extraits du signal), renvoyer un **score continu** et un **niveau de signal** (`low` / `moderate` / `elevated`) dans le format `PredictionResult` standard, prêt pour une **fusion tardive** côté application multimodale.

Ce n’est **pas** un diagnostic médical.

## 2. Dataset utilisé

- **Jeu** : Oxford Parkinson’s Disease Detection (UCI), fichiers `parkinsons.data` et `parkinsons.names`.
- **Copie de travail** : `data/` à la racine de ce dossier.
- **Lignes** : une ligne = un enregistrement vocal résumé par des nombres ; plusieurs lignes par sujet environ.

## 3. Signification des colonnes principales

| Mesure | Idée simple |
|--------|-------------|
| **Jitter** (ex. `MDVP:Jitter(%)`, `RAP`, `PPQ`) | Micro-variations **rapides** de la fréquence fondamentale (instabilité « hauteur »). |
| **Shimmer** (ex. `MDVP:Shimmer`, `APQ*`) | Variations **d’amplitude** (volume). |
| **HNR** (Harmonics-to-Noise Ratio) | Plus le rapport harmoniques/bruit est favorable, plus la voix est « nette ». `NHR` est lié au bruit. |
| **RPDE** / **PPE** / **DFA** | Descripteurs **non linéaires** / complexité (définitions mathématiques avancées ; utiles au modèle comme résumés statistiques du signal). |

Les colonnes `MDVP:Fo(Hz)` et voisines décrivent la **fréquence fondamentale** et ses extrêmes.

## 4. Pourquoi des modèles tabulaires ?

Les entrées sont un **vecteur numérique fixe** (22 features). Les algorithmes tabulaires classiques (régression logistique, SVM, forêts, boosting) sont adaptés, rapides à entraîner et simples à servir avec `joblib`.

## 5. Pourquoi pas encore d’analyse audio brut ?

L’audio brut demande **segmentation**, **extraction de descripteurs** (MFCC, etc.), gestion des formats et du bruit. Ici on réutilise les mesures **déjà publiées** avec le dataset pour se concentrer sur le **protocole ML** (validation, fuite de données, comparaison de modèles, contrat API).

## 6. Modèles testés

- **Régression logistique** : baseline interprétable.
- **SVM RBF** : frontière non linéaire ; utile sur petits jeux ; risque de surapprentissage.
- **Random Forest** : ensemble d’arbres ; importances de variables.
- **ExtraTrees** : proche de RF, plus aléatoire ; souvent stable.
- **Gradient Boosting** : arbres séquentiels ; puissant.
- **XGBoost** : si le paquet est installé ; sinon ignoré sans erreur.

**VotingClassifier** (ensemble de modèles) : à n’ajouter que si plusieurs modèles sont **complémentaires** et qu’un vote améliore nettement la validation.

## 7. Méthode de validation

- **GroupKFold** par **groupe patient** approximé depuis `name` (retrait du suffixe numérique).
- **Glossaire — data leakage** : mélanger des enregistrements du même patient dans train et test surestime la performance.
- **Glossaire — GroupKFold** : chaque fold garde les groupes entiers d’un côté ou de l’autre.
- **Sélection** : on ne retient pas seulement le meilleur **ROC-AUC** moyen ; on pénalise un peu la **variance** entre folds (`mean_auc - 0.35 * std` dans le script) pour favoriser un modèle **plus stable**.

**Glossaire — ROC-AUC** : capacité à classer les Parkinson au-dessus des contrôles sur le score continu ; utile pour comparer des modèles.

### Augmentation tabulaire (SMOTE, optionnelle)

**SMOTE** (*Synthetic Minority Over-sampling Technique*, paquet `imbalanced-learn`) crée des exemples **synthétiques** de la classe minoritaire en interpolant entre voisins dans l’espace des features. Ce n’est **pas** de l’augmentation audio.

- `--augment-cv` : SMOTE sur chaque **train** de pli ; le **test** reste réel.
- `--augment-final` : fit final sur `X, y` suréchantillonnés (le `joblib` sert alors un modèle entraîné aussi sur des synthétiques).

- **Inférence** : inchangée ; elle ne dépend pas de SMOTE au runtime.
- **Code SMOTE** : `src/modalities/voice/augmentation.py` (`apply_smote_safe`).
- **Limites** : petit `n` → synthétiques parfois irréalistes ; ne remplace pas de nouveaux enregistrements réels.

Les métriques exportées dans le `joblib` incluent `metrics.augmentation` (`smote_cv`, `smote_final`) pour la traçabilité.

## 8. Modèle retenu

Après exécution de `scripts/train_voice_tabular.py`, le modèle choisi est indiqué dans `artifact["model"]` (souvent **ExtraTrees** sur ce jeu). Les métriques de leaderboard sont dans `artifact["metrics"]` (y compris les drapeaux `augmentation.smote_cv` / `smote_final` si SMOTE a été utilisé). Le fichier produit est **`models/voice_parkinsons_tabular.joblib`**, chargé par `VoicePredictor`.

## 9. Endpoint `/api/voice/predict`

- **Méthode** : `POST`
- **Content-Type** : `application/json`
- **Corps** : objet contenant soit les clés **exactes** des colonnes features, soit un sous-objet `"features"`, soit des **alias** en snake_case (voir `VOICE_PAYLOAD_ALIASES` dans `src/modalities/voice/features.py`).

Le blueprint Flask est défini dans `src/modalities/voice/routes.py`. Il peut être enregistré par n’importe quelle application Flask qui ajoute la racine de ce dossier au `sys.path`.

## 10. Exemple de payload JSON

```json
{
  "features": {
    "MDVP:Fo(Hz)": 197.076,
    "MDVP:Fhi(Hz)": 206.896,
    "MDVP:Flo(Hz)": 192.055,
    "MDVP:Jitter(%)": 0.00289,
    "MDVP:Jitter(Abs)": 0.00001,
    "MDVP:RAP": 0.00166,
    "MDVP:PPQ": 0.00168,
    "Jitter:DDP": 0.00498,
    "MDVP:Shimmer": 0.01098,
    "MDVP:Shimmer(dB)": 0.097,
    "Shimmer:APQ3": 0.00563,
    "Shimmer:APQ5": 0.0068,
    "MDVP:APQ": 0.00802,
    "Shimmer:DDA": 0.01689,
    "NHR": 0.00339,
    "HNR": 26.775,
    "RPDE": 0.422229,
    "DFA": 0.741367,
    "spread1": -7.3483,
    "spread2": 0.177551,
    "D2": 1.743867,
    "PPE": 0.085569
  }
}
```

## 11. Exemple de réponse `PredictionResult`

```json
{
  "modality": "voice",
  "status": "ok",
  "score": 0.42,
  "confidence": 0.78,
  "label": "moderate",
  "details": {
    "threshold": 0.5,
    "model_path": ".../Voice_Casin_Wayne_Oxford_AI/models/voice_parkinsons_tabular.joblib",
    "model_name": "extra_trees",
    "feature_count": 22
  },
  "warnings": []
}
```

En cas d’erreur, `status` vaut `"error"`, `score` est `null`, `label` est `null`.

**Score vs confidence**

- **Score** : sortie du modèle (probabilité estimée d’être étiqueté Parkinson **dans ce dataset**).
- **Confidence** : **confiance technique** (prudence, qualité des entrées) ; **pas** une probabilité clinique. Dans le code elle est **plafonnée** (jamais 1.0).

## 12. Limites scientifiques

- Jeu **petit** et **ancien** ; pas représentatif de toutes les populations.
- **Pas d’audio** : on ne valide pas la chaîne d’acquisition micro → descripteurs.
- **Scores** : peuvent être **sur-optimistes** si la validation n’est pas respectée (d’où GroupKFold).
- **Probabilités** : pas forcément **calibrées** au sens clinique.

## 13. Évolutions possibles

- Upload audio + extraction MFCC / OpenSMILE + nouveau pipeline.
- Calibration des probabilités (Platt, isotonic) sur un jeu indépendant.
- **LightGBM** / **CatBoost** si besoin, toujours avec validation par groupe.

---

## À retenir simplement

La modalité voix ne prend pas encore un fichier audio. Elle prend des **mesures vocales déjà calculées**. Le modèle apprend à associer certaines variations de ces mesures avec un profil contrôle ou Parkinson **dans ce dataset**. Le résultat est un **score exploratoire**, pas un diagnostic.

---

## Fusion tardive

`src/common/fusion.py` expose `late_fusion(predictions, weights=None)` qui accepte une liste de `PredictionResult`. Les modalités avec `status != "ok"` sont **ignorées**. Les poids par défaut sont dans `DEFAULT_WEIGHTS` ; un objet `weights` peut les surcharger.

---

## Glossaire rapide

| Terme | Explication courte |
|-------|---------------------|
| Overfitting | Le modèle « apprend par cœur » le bruit du train et mal généralise. |
| Pipeline | Chaîne sklearn (ex. normalisation + classifieur) réutilisable à l’inférence. |
| Feature importance | Score indiquant quelles variables influencent le plus un modèle à base d’arbres. |
| Calibration | Ajuster les scores pour qu’ils reflètent des fréquences observables. |
