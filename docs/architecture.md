# Architecture détaillée — `Voice_Casin_Wayne_Oxford_AI`

Ce document décrit **chaque fichier** du projet : son rôle, ce qu’il contient, et ses dépendances vis-à-vis du reste du code. Le projet est **autonome** : entraînement, modèle, prédicteur, Flask blueprint, front, tests et docs vivent tous ici.

## 1. Vue d’ensemble

```text
Voice_Casin_Wayne_Oxford_AI/
├── README.md                        guide rapide installation / entraînement / réutilisation
├── requirements-train.txt           dépendances Python
├── data/                            dataset brut (Oxford / UCI)
├── models/                          artefact joblib produit par l’entraînement
├── scripts/                         scripts CLI (entraînement)
├── src/                             code Python réutilisable
│   ├── common/                      contrat de prédiction + fusion tardive
│   └── modalities/voice/            modalité voix (features, augmentation, prédicteur, routes Flask)
├── app/                             front Flask (template + JS + CSS) liés à la modalité voix
├── notebooks/voice/                 notebook pédagogique d’entraînement
├── docs/                            documentation (ce dossier)
├── tests/                           tests pytest
└── parkinson data-set  cansin wayne oxford/   sauvegarde brute du dataset (référence)
```

## 2. Fichiers à la racine

### `README.md`

Point d’entrée du projet. Décrit :

- L’**objectif** : centraliser tout ce qui est lié à la modalité voix tabulaire.
- La **structure** des dossiers en un coup d’œil.
- Les **commandes** principales : créer le venv, installer `requirements-train.txt`, lancer l’entraînement, exécuter les tests, ouvrir le notebook.
- Le snippet de **réutilisation depuis un autre dépôt** (par exemple l’application multimodale clavier voisine), via `sys.path.insert(0, racine_voice_casin)` puis `from src.modalities.voice.routes import voice_bp`.

À mettre à jour si on ajoute un dossier ou un script important.

### `requirements-train.txt`

Liste des paquets Python nécessaires pour faire tourner le projet de bout en bout :

| Paquet | Pourquoi |
|--------|----------|
| `pandas` | Charger `parkinsons.data` (CSV) et regrouper par patient. |
| `numpy` | Vecteurs de features et calculs numériques. |
| `scikit-learn` | Modèles, pipelines, `GroupKFold`, `StandardScaler`, métriques. |
| `joblib` | Sérialisation / désérialisation de l’artefact modèle. |
| `xgboost` | Modèle candidat supplémentaire (ignoré silencieusement s’il est absent). |
| `imbalanced-learn` | SMOTE (`apply_smote_safe`). |
| `flask` | Servir le blueprint `voice_bp` (`/voice`, `/api/voice/predict`). |
| `pytest` | Lancer les tests dans `tests/`. |

## 3. `data/` — dataset Oxford / UCI

### `data/parkinsons.data`

Le fichier CSV brut du dataset Oxford / UCI. Chaque ligne est un enregistrement vocal résumé par 22 biomarqueurs numériques (Jitter, Shimmer, HNR, RPDE, DFA, spread1/2, D2, PPE, …) plus une colonne `name` (identifiant patient + suffixe `_n`) et `status` (0 = contrôle, 1 = Parkinson). Lu par le script d’entraînement et le notebook.

### `data/parkinsons.names`

Documentation officielle du jeu UCI : description des colonnes, citations à utiliser dans un rapport académique. Référencé dans `docs/voice.md` et le rapport ML.

### `parkinson data-set  cansin wayne oxford/`

Copie miroir des deux fichiers ci-dessus, conservée sous le nom d’origine du téléchargement (référence historique). **Le code ne lit pas ce dossier.** Il peut être supprimé sans rien casser, mais il est gardé pour montrer la provenance.

## 4. `src/` — code Python réutilisable

### `src/__init__.py`

Marqueur de paquet. Permet `from src.xxx import …` à partir du moment où la racine du projet est dans le `sys.path` (cf. `tests/conftest.py` et le script d’entraînement).

### `src/common/__init__.py`

Marqueur de sous-paquet. Aucune logique : il rend `src.common` importable.

### `src/common/schemas.py`

Contrat de données **partagé**, identique à celui du dépôt multimodal côté clavier. Il contient :

- `PredictionStatus = Literal["ok", "insufficient_data", "error"]`
- `RiskLabel = Literal["low", "moderate", "elevated"]`
- `@dataclass PredictionResult` (`modality`, `status`, `score`, `confidence`, `label`, `details`, `warnings`, + `to_dict()`). C’est ce que **renvoie** `VoicePredictor.predict()` et ce que les routes Flask sérialisent en JSON.
- `@dataclass FusionResult` (utilisé par `fusion.py` ; champ supplémentaire : `used_modalities`, `ignored_modalities`).
- `score_to_label(score, low_threshold=0.35, high_threshold=0.58)` : transforme un score continu 0–1 en libellé `low / moderate / elevated`.

Garder ce fichier **synchronisé** avec celui du dépôt clavier si on veut que la fusion fonctionne quand on les recombine.

### `src/common/fusion.py`

Fusion tardive de plusieurs `PredictionResult` (clavier + voix + dessin par exemple).

- `DEFAULT_WEIGHTS = {"keyboard": 1.0, "voice": 1.0, "drawing": 1.0}` : poids par défaut par modalité.
- `_as_prediction_result(raw)` : accepte aussi bien une dataclass que le dict JSON déjà sérialisé.
- `late_fusion(predictions, weights=None) -> FusionResult` :
  - ignore les prédictions dont `status != "ok"` ou `score is None`,
  - pondère chaque score par `weight × confidence`,
  - renvoie un `FusionResult` avec score global, confiance, modalités utilisées / ignorées, et avertissements concaténés.

Ce module est nécessaire dès qu’on veut combiner la voix avec une autre modalité hors de ce projet. Il n’est appelé nulle part en interne (la voix ne se fusionne pas avec elle-même), mais il est testé par `tests/test_fusion_voice.py` pour figer le contrat.

### `src/modalities/__init__.py`

Marqueur de sous-paquet. Pas de logique.

### `src/modalities/voice/__init__.py`

Marqueur du paquet `voice`. Pas de logique.

### `src/modalities/voice/features.py`

Validation et mise en forme des **22 biomarqueurs** envoyés à l’API ou utilisés à l’entraînement.

- `VOICE_FEATURE_COLUMNS` : tuple des 22 noms de colonnes attendus (ordre du dataset).
- `VOICE_PAYLOAD_ALIASES` : alias `snake_case` (par ex. `mdvp_fo_hz` → `MDVP:Fo(Hz)`) pour accepter des clés JSON plus simples côté client.
- `_resolve_payload_keys(payload)` : aplatit `{"features": {...}}` ou un payload à la racine, applique les alias, ignore les champs `name` / `session_id`.
- `build_voice_feature_row(payload, expected_columns)` : retourne `(numpy_row_1xN, missing_keys, warnings)`. Renvoie `(None, …)` si une valeur n’est pas numérique ou si une colonne manque, et inclut une garde contre `NaN`/`inf`.

Utilisé par : `predictor.py` (inférence), `routes.py` (page de saisie), et indirectement par `notebooks/voice/...ipynb` et `scripts/train_voice_tabular.py` pour figer l’ordre des colonnes.

### `src/modalities/voice/augmentation.py`

Implémente `apply_smote_safe(x_train, y_train, *, random_state=42)` :

- Si `imbalanced-learn` n’est pas installé → renvoie l’entrée inchangée avec `ok=False`.
- Si une seule classe est présente, ou si la minoritaire a moins de 2 exemples → idem (`ok=False`).
- Sinon, applique `SMOTE` avec un `k_neighbors` calculé pour rester compatible avec la taille de la minoritaire (`min(5, n_minor - 1)`), retourne `(x_res, y_res, True)`.
- Retombe sur l’entrée d’origine si SMOTE lève `ValueError` (cas pathologique).

Cette fonction est **uniquement** appelée à l’entraînement (`scripts/train_voice_tabular.py`, options `--augment-cv` et `--augment-final`) ou dans le notebook pédagogique. Elle n’est **jamais** appelée à l’inférence : l’augmentation ne s’applique pas aux requêtes utilisateurs.

### `src/modalities/voice/predictor.py`

Le **prédicteur** chargé en runtime.

- `PROJECT_ROOT = Path(__file__).resolve().parents[3]` (= la racine de ce dépôt, soit `Voice_Casin_Wayne_Oxford_AI/`).
- `DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "voice_parkinsons_tabular.joblib"` (chemin **local** à ce projet ; plus aucune référence au dossier multimodal).
- `_technical_confidence(*, completeness, extra_penalty=1.0)` : confiance technique plafonnée à 0.9, jamais 1.0 (prudence pédagogique).
- `class VoicePredictor` :
  - `__init__(model_path, threshold=0.5)` : configure mais ne charge pas l’artefact.
  - `_load_artifact()` : charge le `joblib` une seule fois et met à jour `threshold` depuis l’artefact (cache mémoire).
  - `predict(payload)` : pipeline complet :
    1. validation du payload (type dict, non vide),
    2. existence du fichier modèle (sinon `status="error"`),
    3. extraction du vecteur de features via `build_voice_feature_row`,
    4. appel `predict_proba` (ou `decision_function` + sigmoïde, ou `predict` clampé) selon ce que le pipeline expose,
    5. construction d’un `PredictionResult` avec score, label, confidence, et détails (`threshold`, `model_path`, `model_name`, `feature_count`).

C’est ce que `routes.py` instancie au boot de Flask.

### `src/modalities/voice/routes.py`

Blueprint Flask exposant les deux endpoints voix.

- `voice_bp = Blueprint("voice", __name__)` ;
- `predictor = VoicePredictor()` (instancié une fois par worker Flask) ;
- `GET /voice` → rend `voice.html` avec la liste des 22 colonnes (`feature_columns=list(VOICE_FEATURE_COLUMNS)`) ;
- `POST /api/voice/predict` → lit le JSON, appelle `predictor.predict(payload)`, sérialise `result.to_dict()` en JSON, code 200 pour `ok` / `insufficient_data`, 500 pour `error`.

Réutilisable depuis n’importe quelle app Flask qui :

1. ajoute la racine `Voice_Casin_Wayne_Oxford_AI/` au `sys.path` ;
2. enregistre `voice_bp` ;
3. expose `Voice_Casin_Wayne_Oxford_AI/app/templates/` comme `template_folder` (et `app/static/` comme `static_folder`).

## 5. `scripts/` — entrée CLI d’entraînement

### `scripts/train_voice_tabular.py`

Script principal d’entraînement. Exécution : `python scripts/train_voice_tabular.py` depuis la racine du projet.

Étapes (dans l’ordre du fichier) :

1. **Configuration** :
   - `VOICE_PROJECT_ROOT = Path(__file__).resolve().parents[1]` ajouté au `sys.path` pour rendre `src.modalities.voice.augmentation` importable même sans installer le paquet.
   - `DATA_PATH = VOICE_PROJECT_ROOT / "data" / "parkinsons.data"`.
   - `MODEL_OUT = VOICE_PROJECT_ROOT / "models" / "voice_parkinsons_tabular.joblib"`.
2. **`patient_group(name)`** : déduit le groupe patient depuis `phon_RXX_SXX_X` en supprimant le suffixe numérique (`_\d+$`). Sert à empêcher la fuite de données dans `GroupKFold`.
3. **`build_candidates()`** : six modèles testés en parallèle (régression logistique + SVM RBF dans un `Pipeline` avec `StandardScaler`, plus Random Forest, ExtraTrees, Gradient Boosting, et XGBoost si `xgboost` est installé).
4. **`main()`** :
   - parse `--augment-cv` et `--augment-final`,
   - lit le CSV, sépare features / target / groupes,
   - boucle `GroupKFold(n_splits=5)` × candidats :
     - applique SMOTE sur le train du pli si `--augment-cv`,
     - fit `clone(estimator)`,
     - récupère `predict_proba(...)[:, 1]` (ou sigmoïde de `decision_function`),
     - calcule `roc_auc_score` sur le test (toujours **brut**, jamais SMOTE),
   - sélectionne le meilleur modèle par `mean_auc - 0.35 * std` (pénalise la variance pour préférer un modèle stable),
   - refit final sur tout le jeu (avec SMOTE si `--augment-final`),
   - sérialise un `artifact` dans `models/voice_parkinsons_tabular.joblib`. Contenu :

```python
{
    "pipeline": <model_sklearn>,
    "features": [...22 col...],
    "model": "extra_trees",
    "threshold": 0.5,
    "note": "Modèle tabulaire entraîné sur biomarqueurs Oxford (...)",
    "metrics": {
        "leaderboard": [{"model": ..., "mean_roc_auc": ..., "std_roc_auc": ..., "selection_score": ...}, ...],
        "chosen": "extra_trees",
        "mean_roc_auc": 0.829,
        "std_roc_auc": 0.142,
        "augmentation": {"smote_cv": bool, "smote_final": bool},
    },
    "data_path_hint": "<chemin>/parkinsons.data",
}
```

À l’écran, le script imprime le `leaderboard` JSON puis le chemin du fichier produit.

## 6. `app/` — assets Flask

> Le projet ne contient pas de `create_app()` propre : il fournit les **morceaux** dont une application Flask hôte a besoin. Une app voisine (par ex. l’app clavier) peut s’en servir comme dossiers `template_folder` et `static_folder` supplémentaires.

### `app/templates/voice.html`

Template Jinja qui hérite de `base.html` (présent dans l’app Flask hôte). Affiche :

- un en-tête expliquant qu’on envoie des **biomarqueurs tabulaires**, pas un fichier audio ;
- deux boutons « Charger un exemple contrôle / Parkinson » (raccourcis pédagogiques) ;
- un bouton « Prédire » ;
- une **grille de 22 champs** (boucle Jinja sur `feature_columns`) avec `id="vf-{n}"` et `data-voice-feature="<colonne>"` pour pouvoir être lus côté JS ;
- un panneau résultat (statut, label de signal, score, confiance, modèle, détails JSON, warnings) initialement masqué.

### `app/static/js/voice_predict.js`

Logique navigateur de la page `/voice`.

- `VOICE_EXAMPLES` : deux échantillons réels (un contrôle, un Parkinson) extraits de `parkinsons.data`, utilisés par les boutons « exemple ».
- `fillForm(example)` : remplit les inputs en s’appuyant sur `data-voice-feature`.
- `collectPayload()` : construit `{ features: {...} }` à partir des champs non vides (convertit les `,` en `.`).
- `renderResult(result)` : affiche `status` (en français), `label`, `score`, `confidence`, `details`, `warnings`.
- `storeVoiceResult(result)` : enregistre le résultat dans `sessionStorage` sous la clé `parkinson_result_voice` pour qu’une éventuelle page « Résultat global » d’une app hôte puisse le récupérer.
- Listeners sur les boutons + appel `fetch("/api/voice/predict", { method: "POST", body: JSON.stringify(payload) })`.

### `app/static/css/voice.css`

Styles **spécifiques** à la grille de saisie des 22 biomarqueurs (`.voice-features-grid`, son `label`, son `input`). Les styles génériques (boutons, panneaux, layout) restent côté app hôte pour rester cohérents avec les autres modalités. À inclure dans le template hôte si on veut une mise en page jolie.

## 7. `notebooks/` — exploration pédagogique

### `notebooks/voice/02_train_voice_model_conforme.ipynb`

Notebook qui reproduit pas à pas l’entraînement du script :

- charge `data/parkinsons.data` avec un chemin **relatif** au notebook (`_REPO_ROOT = Path("../..")`, puis `_REPO_ROOT / "data" / "parkinsons.data"`),
- ajoute la racine du projet au `sys.path` pour pouvoir faire `from src.modalities.voice.augmentation import apply_smote_safe`,
- compare les candidats avec `GroupKFold`,
- montre la courbe ROC, les importances de features, et écrit (optionnellement) un `joblib` sous `_REPO_ROOT / "models"`.

Utile pour expliquer le **protocole** ML (fuite de données, GroupKFold, sélection avec pénalité de variance) ; le script CLI est la version « machine » du même flux.

## 8. `docs/` — documentation

### `docs/architecture.md`

**Le fichier que vous lisez.** Description exhaustive de chaque fichier du projet.

### `docs/voice.md`

Documentation **fonctionnelle** de la modalité :

- objectif, dataset, signification des colonnes (Jitter, Shimmer, HNR, …),
- pourquoi des modèles tabulaires et pas encore d’audio brut,
- modèles testés, méthode de validation (GroupKFold), pénalité de variance dans la sélection,
- section SMOTE (`--augment-cv`, `--augment-final`),
- contrat d’API `/api/voice/predict` (payload et réponse type),
- limites scientifiques, évolutions possibles, glossaire.

### `docs/entrainement_voix.md`

Mémo **opérationnel** très court : commandes à taper pour installer, entraîner, et où atterrit le `joblib`. Pratique pour quelqu’un qui ouvre le repo et veut juste lancer la chaîne.

### `docs/rapport_resultats_ml_voix.md`

Rapport **académique** des résultats du modèle voix : tableau de scores (mean / std AUC par modèle), interprétation, choix retenu (ExtraTrees), discussion des limites. Sert de base à la partie « voix » d’un rapport de projet.

## 9. `tests/` — tests unitaires

### `tests/conftest.py`

Ajoute la racine du projet (`Voice_Casin_Wayne_Oxford_AI/`) au `sys.path` **avant** que pytest n’importe les fichiers de test. Indispensable pour que `from src.xxx import …` fonctionne sans installer le paquet.

### `tests/test_augmentation.py`

Deux tests sur `apply_smote_safe` :

1. `test_smote_skipped_single_class` : entrée à une seule classe → la fonction renvoie `ok=False` sans modifier les tableaux.
2. `test_smote_balances_minority_when_imblearn_available` : skipé si `imblearn` n’est pas installé ; sinon, vérifie qu’un déséquilibre 20/5 est rééquilibré et que la classe minoritaire est suréchantillonnée.

### `tests/test_voice_predictor.py`

Six tests sur `VoicePredictor` et `build_voice_feature_row` :

- `test_voice_model_file_exists` / `test_predict_full_payload_ok` / `test_predict_incomplete_payload_error` / `test_predict_payload_not_dict` : skipés automatiquement si `models/voice_parkinsons_tabular.joblib` n’existe pas (entraînement pas encore lancé). Ils vérifient que la prédiction renvoie un score dans `[0,1]`, une confiance `< 1.0`, et un label parmi `low / moderate / elevated` quand tout est OK ; et qu’elle renvoie `status="error"` pour les payloads invalides.
- `test_build_voice_feature_row_missing` : sans payload, toutes les colonnes sont signalées manquantes.
- `test_predict_empty_payload` : modèle introuvable + payload vide → erreur.

### `tests/test_fusion_voice.py`

Deux tests sur `late_fusion` :

1. Une prédiction voix `status="error"` est **ignorée** ; le clavier reste utilisé.
2. Avec clavier et voix tous deux `status="ok"`, les deux apparaissent dans `used_modalities` et le score est dans `[0, 1]`.

Ces tests figent le contrat de fusion : utile pour ne pas casser l’intégration côté app hôte.

## 10. `models/`

### `models/voice_parkinsons_tabular.joblib`

Artefact binaire produit par `scripts/train_voice_tabular.py`. Contenu (cf. section 5). Chargé en mémoire par `VoicePredictor._load_artifact()` à la première requête `/api/voice/predict` puis mis en cache pour les suivantes.

Ce fichier est régénéré **à chaque exécution** du script.

## 11. Cycle de vie typique

1. **Cloner / récupérer** le projet, créer un venv, installer `requirements-train.txt`.
2. **Vérifier** les données : `data/parkinsons.data` et `data/parkinsons.names` doivent être présents.
3. **Entraîner** : `python scripts/train_voice_tabular.py` (optionnellement `--augment-cv`).
4. **Tester** : `python -m pytest tests/` — 10 tests doivent passer (les tests prédicteur skip si le joblib n’est pas encore là).
5. **Servir** :
   - soit une mini app Flask locale qui n’enregistre que `voice_bp` (à écrire en quelques lignes au-dessus de `routes.py`),
   - soit l’app multimodale voisine, qui peut ajouter ce dépôt à son `sys.path` et importer `from src.modalities.voice.routes import voice_bp` ainsi que les `template_folder` / `static_folder` de `app/`.

## 12. Que faire si on déplace un fichier

| Tu déplaces… | Tu dois mettre à jour… |
|--------------|------------------------|
| `src/modalities/voice/augmentation.py` | l’import dans `scripts/train_voice_tabular.py`, `notebooks/voice/...ipynb`, `tests/test_augmentation.py`. |
| `src/modalities/voice/predictor.py` | `routes.py`, `tests/test_voice_predictor.py`, et la valeur de `parents[3]` (calcul de `PROJECT_ROOT`) si la profondeur du fichier change. |
| `src/common/schemas.py` | `predictor.py`, `fusion.py`, `tests/test_fusion_voice.py`. |
| `models/voice_parkinsons_tabular.joblib` | `DEFAULT_MODEL_PATH` dans `predictor.py`. |
| `data/parkinsons.data` | `DATA_PATH` dans `scripts/train_voice_tabular.py` et le `_REPO_ROOT` du notebook. |

## 13. Hors périmètre

- **Pas d’audio brut** : il n’y a aucune extraction de MFCC ni de lecture de `.wav`. Toute la chaîne part de biomarqueurs déjà calculés. Ajouter un upload audio impliquerait un nouveau module `audio_features.py` à côté de `features.py` puis un nouvel artefact `joblib`.
- **Pas de fusion en interne** : la fusion (`src/common/fusion.py`) est fournie comme bibliothèque, mais elle est **utilisée par l’app hôte**, pas appelée depuis ce projet.
- **Pas de mécanisme de versioning du modèle** : on écrase `models/voice_parkinsons_tabular.joblib` à chaque entraînement. Si tu veux comparer deux versions, copie-le manuellement sous un nom différent avant de relancer.
