# Notebooks dessin

Ce dossier contient les expérimentations de la modalité dessin.

## Notebook principal

```text
drawing_model_training.ipynb
```

Ce notebook :

- télécharge les données NewHandPD ;
- prépare les images de spirales ;
- extrait des features HOG ;
- entraîne un pipeline scikit-learn ;
- exporte `models/drawing_spiral_v1_pipeline.joblib`.
