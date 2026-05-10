# Parkinson Detection Multimodale

Prototype hackathon sous forme de petit site web pour tester plusieurs modalités de détection exploratoire de signes compatibles avec Parkinson.

Ce projet ne fournit pas de diagnostic médical. Les résultats sont destinés à la démonstration, à la prévention exploratoire et à la comparaison de méthodes.

## Structure

```text
app/                         # Flask, templates, assets web
src/common/                  # contrat PredictionResult, fusion, registre
src/modalities/keyboard/     # features + prediction keyboard dynamics
src/modalities/voice/        # placeholder contrat commun
src/modalities/drawing/      # placeholder contrat commun
models/                      # modèle clavier versionné, autres modèles ignorés
notebooks/                   # expérimentation
docs/                        # notes et documentation
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

## Dataset — Dessin (NewHandPD)

Les images de spirales utilisées pour entraîner la modalité dessin proviennent du dataset **NewHandPD** (UNESP, Brésil).
Elles ne sont pas versionnées dans le dépôt. Pour les télécharger :

```powershell
# Windows PowerShell
python -c "
import requests, zipfile, io
from pathlib import Path

urls = {
    'hc': 'https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewHealthy/HealthySpiral.zip',
    'pd': 'https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewPatients/PatientSpiral.zip',
}
for label, url in urls.items():
    dest = Path('data/handpd') / label
    dest.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(dest)
    print(f'{label}: {len(list(dest.rglob(\"*.*\")))} fichiers extraits')
"
```

Résultat attendu : `data/handpd/hc/HealthySpiral/*.jpg` (280 images) et `data/handpd/pd/PatientSpiral/*.jpg` (248 images).

## Notebook — Entraînement du modèle de dessin

`notebooks/drawing_model_training.ipynb` entraîne et exporte `models/drawing_spiral_v1_pipeline.joblib`.

Lancer le notebook de haut en bas produit l'artefact et valide le pipeline d'inférence.

## Modèles et pipelines

Le modèle clavier principal est versionné dans `models/keyboard_dynamics_neuroqwerty_v2_pipeline.joblib` pour que l’application fonctionne directement. Les conventions du dossier `models/`, les règles de versionnement des modèles légers et les formats recommandés sont décrits dans [`models/README.md`](models/README.md).

Chaque modalité peut utiliser son propre format de modèle (`joblib`, `.pt`, `.keras`, etc.), mais doit exposer un `predictor.py` qui retourne un `PredictionResult` standard.

## Lancer l’application

```bash
flask --app app.main run --debug
```

Pages principales :

- `/` : accueil et liste des modalités.
- `/keyboard` : test clavier dans le navigateur.
- `/results` : résultat global basé sur les modalités déjà réalisées.
- `/api/keyboard/predict` : endpoint JSON pour la prédiction clavier.
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
