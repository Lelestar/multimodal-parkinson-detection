# Voice_Casin_Wayne_Oxford_AI

Projet **autonome** pour la modalité voix tabulaire (biomarqueurs Oxford / UCI, jeu *cansin wayne oxford*) :

- **Données** + **entraînement** + **SMOTE** + **modèle** produit.
- **Code Python** (features, prédicteur, blueprint Flask, fusion).
- **Front Flask** (template `voice.html`, script `voice_predict.js`, CSS `voice.css`).
- **Tests** unitaires.
- **Notebook** pédagogique.

L’application multimodale voisine (`multimodal-parkinson-detection-main`) peut importer ce paquet en ajoutant la racine du dossier au `sys.path`, mais ce dossier suffit à entraîner, tester et servir la voix de bout en bout.

**Pour comprendre ce que fait chaque fichier**, voir [`docs/architecture.md`](docs/architecture.md).

## Structure

```text
data/                                  parkinsons.data, parkinsons.names
models/                                voice_parkinsons_tabular.joblib (généré)
scripts/
  train_voice_tabular.py               entraînement + export joblib (options SMOTE)
src/
  common/
    schemas.py                         PredictionResult, FusionResult, score_to_label
    fusion.py                          late_fusion (combine plusieurs PredictionResult)
  modalities/voice/
    features.py                        VOICE_FEATURE_COLUMNS + build_voice_feature_row
    augmentation.py                    apply_smote_safe
    predictor.py                       VoicePredictor (charge models/...)
    routes.py                          blueprint Flask /voice + /api/voice/predict
app/
  templates/voice.html                 page Flask de saisie des 22 biomarqueurs
  static/js/voice_predict.js           appel /api/voice/predict + rendu résultat
  static/css/voice.css                 styles spécifiques voix
notebooks/voice/                       notebook pédagogique
docs/                                  voice.md, entrainement_voix.md, rapport_resultats_ml_voix.md
tests/                                 tests pytest (augmentation, prédicteur, fusion)
requirements-train.txt                 dépendances entraînement + flask + pytest
```

## Installation

```powershell
cd Voice_Casin_Wayne_Oxford_AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-train.txt
```

## Entraîner et exporter le modèle

```powershell
python scripts/train_voice_tabular.py
```

Options SMOTE :

```powershell
python scripts/train_voice_tabular.py --augment-cv
python scripts/train_voice_tabular.py --augment-cv --augment-final
```

Détails dans `docs/entrainement_voix.md`.

## Notebook

```powershell
jupyter notebook notebooks/voice/02_train_voice_model_conforme.ipynb
```

## Tests

```powershell
python -m pytest tests/
```

Les tests de prédicteur sont **skip** automatiquement si le modèle n’a pas encore été entraîné.

## Réutiliser depuis un autre dépôt (par ex. multimodal)

```python
import sys
from pathlib import Path

VOICE_ROOT = Path(__file__).resolve().parent.parent / "Voice_Casin_Wayne_Oxford_AI"
sys.path.insert(0, str(VOICE_ROOT))

from src.modalities.voice.routes import voice_bp
from src.common.fusion import late_fusion
```

L’app Flask hôte doit aussi pouvoir trouver `app/templates/voice.html` et `app/static/js/voice_predict.js` (ajouter les dossiers `template_folder` / `static_folder` correspondants).
