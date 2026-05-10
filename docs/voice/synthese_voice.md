# Synthèse - Modalité voix

La modalité voix analyse une phonation soutenue de la voyelle `/a/` enregistrée dans le navigateur. Elle produit un score exploratoire compatible avec la fusion multimodale.

Le résultat ne constitue pas un diagnostic médical.

## Données utilisées

Deux pistes expérimentales sont présentes dans les notebooks :

- `notebooks/voice/01_voice_parkinson_uci470_xgb.ipynb` : première expérimentation sur un petit dataset UCI avec features vocales classiques.
- `notebooks/voice/02_voice_pd_speech_features_xgb.ipynb` : version plus récente basée sur un dataset vocal plus adapté et un modèle `XGBoost`.

## Méthode

La méthode intégrée dans l’application utilise :

- un enregistrement micro dans le navigateur ;
- une conversion en WAV mono 16 bits côté navigateur ;
- une extraction de features vocales inspirées des mesures classiques de dysphonie : fréquence fondamentale, jitter, shimmer, HNR/NHR, RPDE, DFA, D2 et PPE ;
- un pipeline scikit-learn sauvegardé avec `joblib`.

Le modèle utilisé par l’application est :

```text
models/voice_parkinson_xgb.joblib
```

Le prédicteur associé est :

```text
src/modalities/voice/predictor.py
```

## Intégration navigateur

Dans l’application :

1. l’utilisateur ouvre `/voice` ;
2. il enregistre une phonation `/a/` d’au moins 5 secondes ;
3. le navigateur convertit l’audio en WAV ;
4. le backend extrait les features vocales ;
5. le pipeline prédit un score ;
6. le résultat est enregistré pour la fusion multimodale.

## Limites

Les principales limites sont :

- forte dépendance à la qualité du microphone et au bruit ambiant ;
- protocole sensible à la durée, au volume et à la stabilité de la phonation ;
- features extraites en local qui peuvent différer de celles des datasets originaux ;
- modèle exploratoire, non clinique ;
- intérêt principal dans une fusion multimodale, pas comme test autonome.
