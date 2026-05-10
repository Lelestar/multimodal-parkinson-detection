# Modalité dessin

Cette modalité analyse le tracé d’une spirale pour produire un score exploratoire compatible avec la fusion multimodale.

Elle ne fournit pas de diagnostic médical.

## Modèle utilisé

```text
models/drawing_spiral_v1_pipeline.joblib
```

Le prédicteur associé est :

```text
src/modalities/drawing/predictor.py
```

## Pipeline navigateur

1. l’utilisateur ouvre `/drawing` ;
2. il trace une spirale sur le canvas ;
3. le navigateur envoie l’image PNG encodée en base64 ;
4. le backend extrait les features HOG ;
5. le pipeline scikit-learn produit un score ;
6. le résultat est conservé pour la fusion multimodale.

## Documentation

Synthèse :

```text
docs/drawing/synthese_drawing.md
```

Notebook :

```text
notebooks/drawing/drawing_model_training.ipynb
```
