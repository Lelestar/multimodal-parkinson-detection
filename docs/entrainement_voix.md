# Entraînement et augmentation — `Voice_Casin_Wayne_Oxford_AI`

Mémo opérationnel court. Pour l’architecture détaillée de chaque fichier, voir [`architecture.md`](architecture.md). Pour le contrat fonctionnel et l’API, voir [`voice.md`](voice.md).

## Données

- Fichiers : `data/parkinsons.data`, `data/parkinsons.names` (jeu Oxford / UCI).

## Installation

```bash
cd Voice_Casin_Wayne_Oxford_AI
python -m venv .venv
. .venv/Scripts/Activate.ps1     # Windows PowerShell
python -m pip install -r requirements-train.txt
```

## Entraînement

```bash
python scripts/train_voice_tabular.py
```

Options SMOTE :

- `--augment-cv` : SMOTE sur chaque jeu d’apprentissage des plis `GroupKFold` ; le test reste **toujours** brut (pas de fuite).
- `--augment-final` : fit final sur `X, y` après SMOTE (le `joblib` reflète alors un modèle entraîné aussi sur des synthétiques — à utiliser consciemment).

## Sortie

- Modèle : `models/voice_parkinsons_tabular.joblib`.
- Chargé par `src/modalities/voice/predictor.py` (`DEFAULT_MODEL_PATH`) et par toute app Flask qui enregistre `src.modalities.voice.routes.voice_bp`.

## Code SMOTE

- Implémentation : `src/modalities/voice/augmentation.py` (`apply_smote_safe`).
- Tests : `tests/test_augmentation.py` (skip si `imbalanced-learn` n’est pas installé).

## Notebook équivalent

```bash
jupyter notebook notebooks/voice/02_train_voice_model_conforme.ipynb
```

Le notebook reproduit le script pas à pas (chemins relatifs, mêmes candidats, mêmes métriques).
