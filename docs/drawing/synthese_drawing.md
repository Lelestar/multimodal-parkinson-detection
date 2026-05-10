# Synthèse - Modalité dessin

La modalité dessin analyse le tracé d’une spirale réalisé dans le navigateur. Elle produit un score exploratoire destiné à être combiné avec les autres modalités du projet.

Le résultat ne constitue pas un diagnostic médical.

## Données utilisées

Le modèle a été entraîné sur le dataset **NewHandPD** de l’UNESP, au Brésil.

Les images sources ne sont pas versionnées dans le dépôt. Le notebook d’entraînement télécharge les spirales depuis les archives publiques :

- `HealthySpiral.zip` pour les contrôles ;
- `PatientSpiral.zip` pour les patients Parkinson.

## Méthode

La méthode actuelle utilise :

- une image de spirale dessinée dans un canvas HTML ;
- un prétraitement image en niveaux de gris ;
- des features `HOG` ;
- un pipeline scikit-learn sauvegardé avec `joblib`.

Le modèle utilisé par l’application est :

```text
models/drawing_spiral_v1_pipeline.joblib
```

Le prédicteur associé est :

```text
src/modalities/drawing/predictor.py
```

## Intégration navigateur

Dans l’application :

1. l’utilisateur ouvre `/drawing` ;
2. il trace une spirale en suivant le guide affiché ;
3. le canvas est converti en image PNG base64 ;
4. le backend extrait les features HOG ;
5. le pipeline prédit un score ;
6. le résultat est enregistré pour la fusion multimodale.

## Notebook associé

Le notebook d’entraînement est :

```text
notebooks/drawing/drawing_model_training.ipynb
```

Il entraîne et exporte :

```text
models/drawing_spiral_v1_pipeline.joblib
```

## Limites

Les principales limites sont :

- dataset limité et différent d’un tracé navigateur ;
- possible écart entre photos de spirales papier et dessins canvas ;
- dépendance au périphérique utilisé : souris, trackpad, stylet ;
- modèle exploratoire, non clinique ;
- intérêt principal dans une fusion multimodale, pas comme test autonome.
