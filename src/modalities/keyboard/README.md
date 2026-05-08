# Modalité clavier - Keyboard Dynamics

Cette modalité analyse les temps de frappe pour produire un score exploratoire lié à des signes moteurs compatibles avec Parkinson.

Elle ne fournit pas de diagnostic médical. Le score clavier est destiné à être combiné avec les autres modalités du projet.

## Méthode retenue

La méthode utilisée dans l’application est :

```text
agg_timing_xgb
```

Elle utilise :

- des segments de frappe ;
- des features temporelles agrégées sur `hold_time` et `flight_time` ;
- un modèle `XGBoost` ;
- une moyenne des probabilités par segment.

Le modèle chargé par l’application est :

```text
models/keyboard_dynamics_neuroqwerty_agg_timing_xgb.joblib
```

Le prédicteur associé est :

```text
src/modalities/keyboard/predictor.py
```

## Pipeline navigateur

Dans l’application Flask :

1. l’utilisateur ouvre `/keyboard` ;
2. il recopie un texte standardisé ;
3. le navigateur capture les événements `keydown` et `keyup` ;
4. le backend reconstruit les frappes valides ;
5. les features `agg_timing_xgb` sont extraites ;
6. le modèle produit un score ;
7. le résultat est enregistré en session pour la fusion multimodale.

La modalité retourne un `PredictionResult` standard compatible avec la fusion tardive.

## Documentation

Synthèse finale :

```text
docs/keyboard/synthese_finale_keyboard_dynamics.md
```

Parcours expérimental :

```text
docs/keyboard/parcours_experimental_keyboard_dynamics.md
```
