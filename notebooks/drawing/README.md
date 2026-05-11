# Notebooks dessin

Ce dossier contient les expérimentations de la modalité dessin.

## Notebook principal

```text
drawing_model_spiral_dataset.ipynb
```

Ce notebook :

- utilise le dataset de spirales papier ;
- prépare les images de spirales ;
- extrait des features HOG + LBP ;
- entraîne un pipeline scikit-learn ;
- exporte `models/drawing_spiral_v2_hog_lbp_pipeline.joblib`.
