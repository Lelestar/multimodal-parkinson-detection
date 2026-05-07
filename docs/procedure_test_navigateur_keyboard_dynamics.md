# Procedure de test navigateur - Keyboard Dynamics Parkinson

Ce document decrit une procedure MVP pour utiliser la modalite **Keyboard Dynamics** dans un navigateur, avec la methode la plus prometteuse obtenue sur NeuroQWERTY v2:

- segmentation des frappes;
- extraction de features enrichies par segment;
- prediction par segment;
- aggregation des probabilites au niveau session/utilisateur;
- affichage d'un score de screening non diagnostique.

## Objectif

Construire un test simple dans le navigateur qui collecte une frappe standardisee, extrait des indicateurs moteurs, puis retourne un score de risque prudent.

Le test ne doit pas etre presente comme un diagnostic medical. Il doit etre formule comme un outil de prevention, de sensibilisation ou de screening exploratoire.

## Resume de la methode retenue

Le notebook `04_keyboard_dynamics_neuroqwerty_v2_segments.ipynb` montre que la meilleure approche testee est:

- decouper les sessions en segments de frappes;
- extraire des features statistiques par segment;
- entrainer un modele tabulaire;
- agreger les probabilites des segments.

Resultat observe en validation groupee par sujet:

- meilleur modele: `HistGradientBoosting`;
- niveau: `segment_mean_agg`;
- F1 macro moyen: environ **0.795**;
- accuracy moyenne: environ **0.800**;
- balanced accuracy moyenne: environ **0.800**;
- seuil exploratoire out-of-fold: **0.58**;
- macro F1 avec ce seuil: environ **0.81**.

Ces chiffres sont encourageants mais restent exploratoires. Le seuil devra etre recalibre sur un vrai jeu de validation avant un usage public.

## Parcours utilisateur propose

1. L'utilisateur ouvre la page de test.
2. Une courte note explique:
   - que le test analyse la dynamique de frappe;
   - que les donnees sont anonymisees;
   - que le resultat n'est pas un diagnostic.
3. L'utilisateur donne son consentement.
4. Le navigateur affiche un texte a recopier.
5. L'utilisateur tape naturellement pendant environ 3 a 5 minutes.
6. Le navigateur capture les evenements clavier.
7. Les donnees sont envoyees au backend ou transformees localement en features.
8. Le modele predit une probabilite par segment.
9. Les probabilites sont agregees en score de session.
10. L'interface affiche un niveau de signal: faible, modere ou eleve.

## Duree et quantite de frappe

La methode actuelle utilise:

- fenetre: **300 frappes**;
- stride: **150 frappes**;
- minimum utile: **300 frappes**;
- mieux: **600 a 900 frappes**;
- objectif MVP: **3 a 5 minutes** de frappe.

Exemples:

| Frappes utilisables | Segments approximatifs |
|---:|---:|
| 300 | 1 |
| 600 | 3 |
| 900 | 5 |
| 1200 | 7 |

Plusieurs segments permettent de lisser le bruit via aggregation des probabilites.

## Texte de test

Il vaut mieux utiliser un texte standardise plutot qu'une frappe libre.

Objectifs:

- reduire la variabilite entre utilisateurs;
- garantir assez de caracteres;
- inclure lettres, espaces et ponctuation simple;
- eviter les symboles rares;
- permettre une comparaison plus stable.

Instruction utilisateur proposee:

```text
Veuillez recopier le texte affiche naturellement, sans chercher a aller trop vite.
Corrigez vos erreurs comme vous le feriez normalement.
Le test prend environ 3 minutes.
```

Le texte doit etre assez long pour produire au moins 600 frappes. Pour un MVP francophone, utiliser un texte en francais simple et neutre.

## Donnees capturees cote navigateur

Le navigateur doit capturer les evenements `keydown` et `keyup`.

Schema logique:

```json
{
  "session_id": "uuid",
  "participant_id": "anonymous-id",
  "started_at": "timestamp",
  "events": [
    {
      "type": "keydown",
      "code": "KeyA",
      "key_category": "letter",
      "key_side": "left",
      "timestamp_ms": 123456.78
    },
    {
      "type": "keyup",
      "code": "KeyA",
      "key_category": "letter",
      "key_side": "left",
      "timestamp_ms": 123532.10
    }
  ]
}
```

Champs utiles:

- `type`: `keydown` ou `keyup`;
- `code`: code clavier navigateur, ex. `KeyA`, `Space`, `Backspace`;
- `key_category`: `letter`, `space`, `punct`, `backspace`, `other`;
- `key_side`: `left`, `right`, `unknown`;
- `timestamp_ms`: timestamp haute resolution via `performance.now()`.

## Confidentialite

Eviter de stocker le texte exact tape si ce n'est pas necessaire.

Options:

1. Capturer uniquement les codes/categorie de touches.
2. Convertir directement les evenements en features.
3. Supprimer les caracteres bruts apres extraction.
4. Ne jamais afficher un diagnostic medical.

La dynamique de frappe peut etre consideree comme une donnee comportementale sensible.

## Nettoyage des evenements

Le backend doit reconstruire des frappes valides:

1. associer chaque `keydown` au `keyup` correspondant;
2. calculer `hold_time = keyup - keydown`;
3. calculer `flight_time = current_keydown - previous_keydown`;
4. retirer les frappes sans `keyup`;
5. retirer les temps negatifs ou aberrants;
6. conserver ou mesurer `Backspace` comme indicateur d'erreur, mais ne pas l'inclure aveuglement dans les touches utiles.

Regles inspirees de NeuroQWERTY:

- retirer les touches meta longues: Shift, Control, Alt;
- retirer souris et evenements non clavier;
- filtrer les hold times impossibles;
- garder les espaces et ponctuations simples comme categories.

## Features par segment

Pour chaque segment de 300 frappes, calculer:

### Volume et rythme

- `n_keystrokes`;
- `duration_sec`;
- `keys_per_min`.

### Hold time

- `mean_hold`;
- `std_hold`;
- `median_hold`;
- `iqr_hold`;
- `q10_hold`;
- `q90_hold`;
- `skew_hold`;
- `kurt_hold`;
- `cv_hold`;
- `long_hold_rate`.

### Flight time

- `mean_flight`;
- `std_flight`;
- `median_flight`;
- `iqr_flight`;
- `q10_flight`;
- `q90_flight`;
- `skew_flight`;
- `kurt_flight`;
- `cv_flight`;
- `long_flight_rate`.

### Ratios et categories

- `hold_to_flight`;
- `space_punct_rate`;
- `left_rate`;
- `right_rate`;
- `hand_switch_rate`;
- `hand_entropy`.

Ces features doivent etre exactement les memes entre le notebook d'entrainement et l'API de prediction.

## Pipeline de prediction

Pipeline logique:

```text
evenements clavier bruts
-> nettoyage
-> reconstruction des frappes
-> decoupage en segments de 300 frappes
-> extraction features par segment
-> prediction par segment
-> aggregation des probabilites
-> score final de session
```

Aggregation actuelle:

```text
score_session = moyenne(probabilites_segments)
```

Decision exploratoire:

```text
signal_eleve si score_session >= 0.58
```

Ce seuil vient du notebook v2 et doit rester considere comme exploratoire.

## Affichage du resultat

Ne pas afficher:

```text
Vous avez Parkinson.
```

Afficher plutot:

```text
Score moteur clavier : faible / modere / eleve
```

Texte de prudence recommande:

```text
Ce resultat indique uniquement un signal dans votre dynamique de frappe.
Il ne constitue pas un diagnostic medical.
Si vous avez des inquietudes, parlez-en a un professionnel de sante.
```

## Architecture MVP

Architecture simple:

```text
Frontend navigateur
  - affiche le texte
  - capture keydown/keyup
  - envoie les evenements

Backend FastAPI ou Flask
  - nettoie les evenements
  - extrait les features
  - charge le modele joblib
  - retourne score + interpretation

Modele
  - HistGradientBoosting segment-level
  - aggregation moyenne par session
```

Pour un hackathon, il est preferable de garder l'extraction de features cote backend afin d'eviter une divergence entre entrainement et inference.

## Structure code recommandee

Extraire la logique des notebooks vers un module partage:

```text
src/
  keyboard/
    events.py        # parsing keydown/keyup
    features.py      # extraction features segments
    predict.py       # chargement modele + prediction
    schema.py        # schemas Pydantic si FastAPI
```

Les notebooks doivent utiliser les memes fonctions que l'API.

## API minimale

Endpoint possible:

```http
POST /api/keyboard/predict
```

Payload:

```json
{
  "session_id": "uuid",
  "events": []
}
```

Reponse:

```json
{
  "session_id": "uuid",
  "n_keystrokes": 842,
  "n_segments": 4,
  "score": 0.63,
  "signal_level": "elevated",
  "disclaimer": "This is not a medical diagnosis."
}
```

## Points de vigilance

- Le modele est entraine sur NeuroQWERTY, pas sur des donnees navigateur modernes.
- Le clavier, la langue, le navigateur et le device peuvent changer les distributions.
- Le seuil `0.58` est exploratoire.
- Les resultats doivent etre presentes comme prevention/screening, pas diagnostic.
- Il faut eviter de conserver du texte sensible.
- Il faudra idealement collecter quelques donnees internes de calibration pendant le hackathon.

## Prochaine etape technique

La prochaine etape logique est de sortir l'extraction de features du notebook v2 vers un module Python partage, puis de creer une petite API locale capable de charger:

```text
models/keyboard_dynamics_neuroqwerty_v2_pipeline.joblib
```

et de retourner un score a partir d'evenements clavier navigateur.

