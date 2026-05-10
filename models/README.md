# Modèles

Ce dossier contient les artefacts nécessaires à l’inférence locale. Le README principal explique l’architecture générale; ce fichier précise les conventions du dossier `models/`.

## Modèles versionnés

```text
models/keyboard_dynamics_neuroqwerty_agg_timing_xgb.joblib
models/drawing_spiral_v1_pipeline.joblib
models/voice_parkinson_xgb.joblib
```

Ces modèles légers sont conservés dans le repo pour que les modalités clavier, dessin et voix fonctionnent directement.

Le modèle clavier utilise des features temporelles agrégées et `XGBoost`. Le modèle dessin utilise des features HOG extraites d’une image de spirale et un pipeline scikit-learn. Le modèle voix utilise des features vocales extraites avec Praat/Parselmouth et un pipeline scikit-learn.

Les modèles légers nécessaires à la démonstration peuvent aussi être ajoutés au repo, à condition qu’ils restent raisonnables en taille et qu’ils soient utiles au lancement local de l’application. Dans ce cas, il faut ajuster `.gitignore` pour autoriser explicitement le fichier concerné.

L’artefact peut contenir, selon la modalité :

- `pipeline` : pipeline scikit-learn chargeable par `joblib`.
- `features` : liste des colonnes attendues.
- `model` : nom court du modèle entraîné.
- `feature_builder` : extracteur de features à utiliser côté application, par exemple `agg_timing_xgb`.
- `threshold` : seuil de décision, actuellement `0.50` pour `keyboard_dynamics_neuroqwerty_agg_timing_xgb.joblib`.
- `note` : contexte d’entraînement.

Certains artefacts contiennent aussi des champs propres à leur modalité, comme `hog_params`, `cv_auc_mean` ou `test_auc` pour le dessin, ou `feature_medians` pour la voix.

## Modèles locaux non versionnés

Les modèles lourds, temporaires ou purement expérimentaux restent ignorés par git pour éviter de versionner des fichiers inutiles ou difficiles à manipuler. Exemples :

```text
models/voice_*.joblib
models/voice_*.pt
models/drawing_*.joblib
models/drawing_*.pt
models/*_experimental.*
```

Chaque modalité doit documenter dans son `predictor.py` le chemin attendu et le format de chargement.

## Convention recommandée

Pour un modèle scikit-learn, utiliser de préférence un artefact `joblib` contenant :

```python
{
    "pipeline": fitted_pipeline,
    "features": feature_names,
    "model": "short_model_name",
    "threshold": threshold,
    "note": "training context"
}
```

Pour un modèle deep learning, utiliser le format natif du framework (`.pt`, `.pth`, `.keras`, SavedModel) et garder le prétraitement dans `src/modalities/<modality>/predictor.py`.

Dans tous les cas, le predictor doit retourner un `PredictionResult` standard pour que la fusion tardive puisse fonctionner.
