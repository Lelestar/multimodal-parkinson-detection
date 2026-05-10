# Modalité voix

Cette modalité analyse une phonation soutenue pour produire un score exploratoire compatible avec la fusion multimodale.

Elle ne fournit pas de diagnostic médical.

## Modèle attendu

```text
models/voice_parkinson_xgb.joblib
```

Le modèle n’est pas versionné dans le merge actuel. Il doit être généré depuis le notebook voix avant d’utiliser cette modalité en local.

Le prédicteur associé est :

```text
src/modalities/voice/predictor.py
```

## Pipeline navigateur

1. l’utilisateur ouvre `/voice` ;
2. il enregistre une phonation `/a/` ;
3. le navigateur convertit l’audio en WAV ;
4. le backend extrait les features vocales ;
5. le pipeline scikit-learn produit un score ;
6. le résultat est conservé pour la fusion multimodale.

## Documentation

Synthèse :

```text
docs/voice/synthese_voice.md
```

Notebook principal :

```text
notebooks/voice/02_voice_pd_speech_features_xgb.ipynb
```
