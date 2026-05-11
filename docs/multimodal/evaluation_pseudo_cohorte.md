# Evaluation multimodale par pseudo-cohorte out-of-fold

Nous n’avons pas trouvé de dataset public contenant, pour les mêmes personnes, les trois modalités du projet : clavier, voix et dessin. L’évaluation multimodale proposée construit donc une pseudo-cohorte cross-dataset.

La version actuelle utilise des scores unimodaux **out-of-fold**. Cela signifie qu’un score utilisé dans la fusion est toujours produit par un modèle entraîné sans l’exemple évalué. Cette étape évite d’évaluer la fusion avec des modèles finaux déjà entraînés sur les mêmes données.

## Principe

Pour chaque classe, le script associe aléatoirement :

- un score de dessin issu du dataset de spirales ;
- un score clavier issu de NeuroQWERTY ;
- un score voix issu du dataset `pd_speech_features`.

Ces éléments ne proviennent pas de la même personne. Ils sont seulement appariés par label : contrôle ou Parkinson.

## Pourquoi un appariement aléatoire ?

Comme les datasets sont séparés, nous n’avons pas de vrais triplets :

```text
patient_001 -> clavier + voix + dessin
```

Nous avons plutôt trois ensembles indépendants :

```text
clavier -> sujets A, B, C...
voix    -> sujets X, Y, Z...
dessin  -> sujets M, N, O...
```

L’appariement aléatoire construit donc des pseudo-patients en respectant uniquement la classe :

```text
pseudo-patient contrôle =
  score clavier contrôle
  + score voix contrôle
  + score dessin contrôle

pseudo-patient Parkinson =
  score clavier Parkinson
  + score voix Parkinson
  + score dessin Parkinson
```

L’intérêt de répéter cet appariement aléatoire est de ne pas dépendre d’un seul ordre arbitraire des fichiers ou des lignes. Un appariement fixe pourrait donner un résultat influencé par le hasard du tri des données. En répétant l’opération plusieurs fois, on estime plutôt le comportement moyen de la fusion tardive.

Cette stratégie est préférable à :

- fusionner les scores dans l’ordre des fichiers, car cet ordre n’a aucun sens médical ;
- moyenner directement chaque modalité, car cela supprime la variabilité individuelle ;
- utiliser les modèles finaux sur tous les exemples, car cela crée une fuite train/test ;
- présenter les sujets comme étant réellement les mêmes, car ce serait méthodologiquement faux.

La question évaluée devient donc :

```text
Si chaque modalité produit des scores out-of-fold raisonnablement séparés,
est-ce que la fusion tardive améliore la séparation globale ?
```

Cette approche reste exploratoire. Elle teste la cohérence de la fusion, mais elle ne remplace pas une validation sur un vrai dataset multimodal collecté chez les mêmes personnes.

## Séparation train/test

L’évaluation est réalisée en deux temps.

1. Génération des scores out-of-fold :

```bash
python scripts/generate_unimodal_oof_scores.py
```

Pour chaque modalité, le script :

- repart des données brutes de la modalité ;
- crée des folds stratifiés ;
- garde les groupes d’un même sujet dans le même fold quand un identifiant sujet est disponible ;
- entraîne le modèle uniquement sur les folds d’entraînement ;
- prédit uniquement sur le fold de test ;
- sauvegarde les scores dans `data/processed/multimodal_oof_scores.csv`.

Les groupes utilisés sont :

- dessin : identifiant extrait du nom de fichier de spirale ;
- clavier : `pID` NeuroQWERTY ;
- voix : colonne `id` du dataset `pd_speech_features`.

2. Evaluation de la fusion :

```bash
python scripts/fusion_dataset_validation.py
```

Le script de fusion ne recharge pas les modèles finaux `.joblib` pour scorer les datasets. Il lit uniquement les scores out-of-fold déjà générés.

## Objectif

Cette évaluation sert à :

- vérifier que la fusion tardive fonctionne avec les trois modalités ;
- comparer les scores individuels, les paires de modalités et la fusion complète ;
- observer si la fusion garde une séparation cohérente entre contrôles et Parkinson.

Elle ne doit pas être présentée comme une performance clinique sur de vrais sujets multimodaux.

## Chemins attendus

Sources des datasets :

- Dessin : [Mendeley Data - Spiral Drawing Dataset](https://data.mendeley.com/datasets/fd5wd6wmdj/1)
- Clavier : [PhysioNet - NeuroQWERTY MIT-CSXPD](https://physionet.org/content/nqmitcsxpd/1.0.0/)
- Voix : [Kaggle - Parkinson's Disease Speech Signal Features](https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features)

```text
data/
  spiral/
    testing/
      healthy/
      parkinson/
  neuroqwerty-mit-csxpd-dataset-1.0.0/
    MIT-CS1PD/
      GT_DataPD_MIT-CS1PD.csv
      data_MIT-CS1PD/
  parkinson_disease_classification/
    pd_speech_features.csv
```

Modèles utilisés :

```text
models/drawing_spiral_v2_hog_lbp_pipeline.joblib
models/keyboard_dynamics_neuroqwerty_agg_timing_xgb.joblib
models/voice_parkinson_xgb.joblib
```

## Scripts

```bash
python scripts/generate_unimodal_oof_scores.py
python scripts/fusion_dataset_validation.py
```

Le premier script génère les scores hors entraînement. Le second répète des appariements aléatoires et rapporte, pour chaque configuration :

- AUC moyenne et écart-type ;
- balanced accuracy moyenne et écart-type ;
- score moyen des contrôles et des patients Parkinson.

## Limites

- Les patients sont composites : les modalités ne viennent pas des mêmes individus.
- Les scores sont out-of-fold, mais les choix de modèles et de features ont été faits à partir d’expérimentations préalables sur ces datasets.
- Les folds sont groupés par sujet quand l’identifiant est disponible, mais les datasets n’ont pas tous la même richesse clinique.
- Les résultats servent à comparer des tendances de fusion, pas à valider médicalement le système.
