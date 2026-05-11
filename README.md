# Parkinson Detection Multimodale

Prototype hackathon sous forme de petit site web pour tester plusieurs modalités de détection exploratoire de signes compatibles avec Parkinson.

Ce projet ne fournit pas de diagnostic médical. Les résultats sont destinés à la démonstration, à la prévention exploratoire et à la comparaison de méthodes.

## Structure

```text
app/                         # Flask, templates, assets web
src/common/                  # contrat PredictionResult, fusion, registre
src/modalities/keyboard/     # features + prediction keyboard dynamics
src/modalities/voice/        # capture et prédiction voix
src/modalities/drawing/      # capture et prédiction drawing dynamics
models/                      # modèles légers versionnés, autres modèles ignorés
notebooks/                   # expérimentations par modalité
notebooks/keyboard/          # notebooks clavier
notebooks/drawing/           # notebooks dessin
notebooks/voice/             # notebooks voix
docs/                        # documentation par modalité
docs/keyboard/               # documentation clavier
docs/drawing/                # documentation dessin
docs/voice/                  # documentation voix
```

## Installation

Linux/macOS :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Avec `uv` :

Linux/macOS :

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Windows PowerShell :

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Pour exécuter les notebooks expérimentaux, installer aussi les dépendances dédiées :

```bash
python -m pip install -r requirements-notebooks.txt
```

Avec `uv` :

```bash
uv pip install -r requirements-notebooks.txt
```

## Datasets

Les datasets ne sont pas inclus dans le dépôt. Il faut les télécharger manuellement et les placer dans les dossiers correspondants.

### Dessin — Spiral dataset

Source : [Mendeley Data – fd5wd6wmdj](https://data.mendeley.com/datasets/fd5wd6wmdj/1)

Télécharger et décompresser dans :

```
data/spiral/
├── training/
│   ├── healthy/
│   └── parkinson/
└── testing/
    ├── healthy/
    └── parkinson/
```

### Clavier — NeuroQWERTY MIT-CS1PD / MIT-CS2PD

Source : [PhysioNet – nqmitcsxpd 1.0.0](https://physionet.org/content/nqmitcsxpd/1.0.0/)

Télécharger et décompresser dans :

```
data/keyboard/
├── MIT-CS1PD/
│   ├── GT_DataPD_MIT-CS1PD.csv
│   └── data_MIT-CS1PD/
│       └── *.csv
└── MIT-CS2PD/
    ├── GT_DataPD_MIT-CS2PD.csv
    └── data_MIT-CS2PD/
        └── *.csv
```

### Voix — Parkinson's Disease Speech Signal Features (Sakar 2019)

Source : [Kaggle – dipayanbiswas/parkinsons-disease-speech-signal-features](https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features)

Télécharger `pd_speech_features.csv` et le placer dans :

```
data/speech/
└── pd_speech_features.csv
```

---

## Modèles et pipelines

Le modèle clavier principal est versionné dans `models/keyboard_dynamics_neuroqwerty_agg_timing_xgb.joblib` pour que l’application fonctionne directement. Les conventions du dossier `models/`, les règles de versionnement des modèles légers et les formats recommandés sont décrits dans [`models/README.md`](models/README.md).

Chaque modalité peut utiliser son propre format de modèle (`joblib`, `.pt`, `.keras`, etc.), mais doit exposer un `predictor.py` qui retourne un `PredictionResult` standard.

## Documentation par modalité

- Clavier : [`docs/keyboard/synthese_finale_keyboard_dynamics.md`](docs/keyboard/synthese_finale_keyboard_dynamics.md)
- Dessin : [`docs/drawing/synthese_drawing.md`](docs/drawing/synthese_drawing.md)
- Voix : [`docs/voice/synthese_voice.md`](docs/voice/synthese_voice.md)

## Lancer l’application

```bash
flask --app app.main run --debug
```

Pages principales :

- `/` : accueil et liste des modalités.
- `/keyboard` : test clavier dans le navigateur.
- `/voice` : test voix dans le navigateur.
- `/drawing` : test dessin dans le navigateur.
- `/results` : résultat global basé sur les modalités déjà réalisées.
- `/api/keyboard/predict` : endpoint JSON pour la prédiction clavier.
- `/api/voice/predict` : endpoint JSON pour la prédiction voix.
- `/api/drawing/predict` : endpoint JSON pour la prédiction dessin.
- `/api/fusion` : fusion tardive de résultats de modalités.

## Contrat de prédiction

Chaque modalité retourne une structure standardisée :

```python
{
    "modality": "keyboard",
    "status": "ok",
    "score": 0.62,
    "confidence": 0.9,
    "label": "elevated",
    "details": {},
    "warnings": []
}
```

La fusion tardive utilise seulement `status`, `score`, `confidence` et un poids par modalité :

```text
score_final = sum(score_i * weight_i * confidence_i) / sum(weight_i * confidence_i)
```

## Workflow Git conseillé

- `main` : socle stable, contrat commun, app qui démarre.
- `feature/keyboard-dynamics` : modalité clavier.
- `feature/voice` : modalité voix.
- `feature/drawing` : modalité dessin.
- `feature/fusion` : intégration fusion si elle évolue beaucoup.

Chaque modalité doit travailler surtout dans `src/modalities/<modality>/` et exposer un endpoint ou un predictor qui retourne `PredictionResult`. Les changements dans `src/common/` doivent rester courts et explicites, car ils impactent toute l’équipe.
