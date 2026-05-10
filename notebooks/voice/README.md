# Notebooks voix

Ce dossier regroupe les expérimentations de la modalité voix.

## Notebooks

- `01_voice_parkinson_uci470_xgb.ipynb` : première baseline sur un petit dataset UCI, utile pour valider le pipeline général.
- `02_voice_pd_speech_features_xgb.ipynb` : expérimentation principale actuelle, avec extraction de features vocales et entraînement du modèle utilisé par l’application.

## Artefact produit

Le notebook principal doit exporter :

```text
models/voice_parkinson_xgb.joblib
```

Cet artefact est utilisé par :

```text
src/modalities/voice/predictor.py
```
