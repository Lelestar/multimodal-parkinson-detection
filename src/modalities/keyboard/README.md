# Modalité clavier - Keyboard Dynamics

Cette branche contient le travail autour de la modalité **Keyboard Dynamics**, c’est-à-dire l’analyse des temps de frappe pour produire un score exploratoire lié à des signes moteurs compatibles avec Parkinson.

Cette modalité ne fournit pas de diagnostic médical. Elle sert uniquement à la démonstration et à l’expérimentation dans le cadre du projet multimodal.

## Idée générale

Le principe est de mesurer la manière dont une personne tape au clavier :

- durée pendant laquelle une touche reste enfoncée (`hold time`) ;
- temps entre deux frappes (`flight time`) ;
- rythme global de frappe ;
- variabilité des timings ;
- répartition approximative gauche/droite des touches ;
- stabilité du signal sur plusieurs segments.

Ces indicateurs sont ensuite passés à un modèle entraîné sur des données de frappe.

## Références et inspiration

Le travail s’inspire principalement :

- de l’article de Padate, Chavan et Abin, **“Parkinson’s Disease Detection Using Keystroke Dynamics with PSO-Based Feature Selection and Ensemble Voting Classifier”**, publié dans les actes de l’ICSIAIML 2025 chez Atlantis Press, DOI [`10.2991/978-94-6463-948-3_19`](https://doi.org/10.2991/978-94-6463-948-3_19) ;
- du dataset **NeuroQWERTY / MIT-CS1PD / MIT-CS2PD**, utilisé comme base principale plus contrôlée ;
- d’expérimentations complémentaires sur **Tappy**, plus volumineux mais moins contrôlé ;
- d’une réflexion exploratoire autour de modèles séquentiels type TypeFormer, sans intégration finale dans l’app pour l’instant.

L’article de Padate et al. annonce des résultats très élevés, mais l’analyse critique du projet a montré plusieurs limites possibles, notamment le risque de fuite de données si les splits ne sont pas groupés par sujet. La validation retenue ici utilise donc une séparation groupée par participant.

## Notebooks associés

Les notebooks de cette branche documentent les essais principaux :

```text
notebooks/01_keyboard_dynamics_neuroqwerty.ipynb
notebooks/02_keyboard_dynamics_tappy.ipynb
notebooks/03_keyboard_dynamics_tappy_sequence_dl.ipynb
notebooks/04_keyboard_dynamics_neuroqwerty_v2_segments.ipynb
```

Le notebook actuellement le plus important pour l’application est :

```text
notebooks/04_keyboard_dynamics_neuroqwerty_v2_segments.ipynb
```

Il introduit :

- segmentation des sessions en fenêtres de frappe ;
- features enrichies ;
- validation groupée par sujet ;
- agrégation des probabilités par session ;
- export du modèle utilisé par l’application.

## Modèle utilisé dans l’application

L’application charge actuellement :

```text
models/keyboard_dynamics_neuroqwerty_v2_pipeline.joblib
```

Cet artefact contient :

- le pipeline scikit-learn entraîné ;
- la liste des features attendues ;
- le type de modèle (`HistGB`) ;
- le niveau d’agrégation (`segment_mean_agg`) ;
- le seuil exploratoire retenu (`0.58`).

Le fichier `joblib` n’est pas le modèle en soi : c’est le format utilisé pour sauvegarder et recharger l’objet Python contenant le pipeline complet.

## Pipeline navigateur

Dans l’application Flask :

1. l’utilisateur ouvre `/keyboard` ;
2. il recopie un texte standardisé ;
3. le navigateur capture les événements `keydown` et `keyup` ;
4. le backend reconstruit les frappes valides ;
5. les frappes sont découpées en segments de `300` frappes ;
6. les features sont extraites par segment ;
7. le pipeline prédit une probabilité par segment ;
8. les probabilités sont moyennées pour produire un score de session ;
9. le résultat est enregistré en session navigateur pour la fusion globale.

Le bouton d’analyse est désactivé tant que le minimum de `300` frappes valides n’est pas atteint.

## Limites importantes

- Le modèle est exploratoire et entraîné sur un petit nombre de sujets.
- Les performances ne doivent pas être interprétées comme une performance clinique.
- Le modèle actuel utilise encore des features dépendantes de la disposition du clavier (`left_rate`, `right_rate`, `hand_switch_rate`, `hand_entropy`).
- Le protocole NeuroQWERTY est plus cohérent avec un clavier QWERTY; les dispositions comme AZERTY peuvent modifier certaines features.
- Une future version pourrait comparer un modèle “timing-only” sans features dépendantes du layout.

## Intégration avec le reste du projet

La modalité clavier retourne un `PredictionResult` standard :

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

La page `/results` peut ensuite fusionner ce résultat avec les futures modalités voix et dessin via la fusion tardive commune.
