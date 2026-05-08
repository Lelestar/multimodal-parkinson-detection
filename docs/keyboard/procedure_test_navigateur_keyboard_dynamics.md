# Procédure de test navigateur - Modalité clavier

Ce document décrit la procédure utilisée par l’application web pour la modalité **Keyboard Dynamics**.

Le test ne fournit pas de diagnostic médical. Il produit un score exploratoire destiné à être combiné avec les autres modalités du projet.

## Objectif du test

Le test navigateur vise à mesurer la dynamique de frappe d’un utilisateur sur un texte standardisé.

Les événements clavier sont utilisés pour calculer des indicateurs temporels comme :

- la durée d’appui sur les touches ;
- le délai entre deux frappes ;
- la vitesse de frappe ;
- la variabilité du rythme ;
- la stabilité du signal sur plusieurs segments.

Ces indicateurs sont ensuite passés au modèle clavier entraîné sur NeuroQWERTY.

## Modèle utilisé

Le modèle actuellement utilisé par l’application est :

```text
models/keyboard_dynamics_neuroqwerty_agg_timing_xgb.joblib
```

La méthode associée est :

```text
agg_timing_xgb
```

Elle repose sur :

- des features temporelles agrégées ;
- un modèle `XGBoost` ;
- une prédiction par segment ;
- une moyenne des probabilités de segments pour obtenir le score final.

Le prédicteur applicatif est :

```text
src/modalities/keyboard/predictor.py
```

## Déroulement utilisateur

1. L’utilisateur ouvre la page `/keyboard`.
2. Les instructions du test sont affichées.
3. L’utilisateur recopie le texte standardisé dans la zone de saisie.
4. Le navigateur capture les événements `keydown` et `keyup`.
5. Une progression indique le nombre de frappes valides capturées.
6. Le bouton d’analyse reste désactivé tant que le minimum de frappes valides n’est pas atteint.
7. Après analyse, l’interface affiche le score clavier, le niveau de signal, la confiance et le nombre de frappes valides.
8. Le score est conservé en session navigateur pour la fusion multimodale.

## Quantité de frappe

Les paramètres actuels sont :

| Paramètre | Valeur |
|---|---:|
| Fenêtre | 300 frappes |
| Stride | 150 frappes |
| Minimum pour analyser | 300 frappes valides |
| Recommandé | 600 frappes valides ou plus |

Exemples :

| Frappes valides | Segments approximatifs |
|---:|---:|
| 300 | 1 |
| 600 | 3 |
| 900 | 5 |
| 1200 | 7 |

Plusieurs segments permettent de lisser le score final.

## Texte à recopier

Le test utilise un texte standardisé plutôt qu’une frappe libre.

L’objectif est de :

- réduire la variabilité entre utilisateurs ;
- garantir une quantité suffisante de frappes ;
- inclure des lettres, espaces et ponctuations simples ;
- éviter les symboles rares ;
- rendre les sessions plus comparables.

L’utilisateur est invité à recopier le texte naturellement, sans chercher à accélérer ou ralentir volontairement sa frappe.

## Capture navigateur

Le navigateur capture les événements suivants :

```json
{
  "type": "keydown",
  "code": "KeyA",
  "key": "a",
  "repeat": false,
  "timestamp_ms": 123456.78
}
```

Champs principaux :

- `type` : `keydown` ou `keyup` ;
- `code` : code clavier navigateur ;
- `key` : touche associée ;
- `repeat` : répétition automatique ou non ;
- `timestamp_ms` : temps haute résolution via `performance.now()`.

L’implémentation actuelle peut aussi envoyer `key_side` et `key_category`, mais ces champs ne sont pas utilisés par le modèle final `agg_timing_xgb`. Ils restent hérités des essais précédents qui utilisaient des features liées au layout clavier, comme `left_rate`, `right_rate` ou `hand_switch_rate`.

Le collage dans la zone de saisie est bloqué pour préserver la dynamique réelle de frappe.

## Nettoyage côté backend

Le backend reconstruit les frappes valides à partir des paires `keydown` / `keyup`.

Les règles principales sont :

- ignorer les événements incomplets ;
- ignorer les répétitions automatiques ;
- ignorer les touches méta longues (`Shift`, `Control`, `Alt`, `Meta`, etc.) ;
- ignorer `Backspace`, `Delete`, `Tab`, `Escape` ;
- filtrer les durées négatives ou aberrantes ;
- calculer `hold_time` et `flight_time`.

Les frappes valides sont ensuite segmentées pour construire les features attendues par le modèle `agg_timing_xgb`.

## Pipeline de prédiction

Pipeline logique :

```text
événements clavier bruts
-> reconstruction des frappes
-> filtrage des frappes valides
-> découpage en segments
-> extraction des features agg_timing_xgb
-> prédiction par segment
-> moyenne des probabilités
-> score clavier final
```

Le seuil de décision du modèle est actuellement :

```text
0.50
```

Ce seuil est chargé depuis l’artefact `joblib`.

## Résultat affiché

L’interface affiche :

- le niveau de signal : faible, modéré ou élevé ;
- le score exploratoire ;
- la confiance du signal ;
- le nombre de frappes valides ;
- les avertissements éventuels.

Le résultat est formulé comme un signal exploratoire, jamais comme un diagnostic.

## Protocole clavier

Le modèle a été entraîné sur NeuroQWERTY. Pour respecter au mieux ce protocole, il est préférable d’utiliser un clavier QWERTY.

Les dispositions comme AZERTY peuvent modifier certaines mesures liées à la position des touches. Le modèle final actuel limite ce risque en utilisant principalement des features temporelles, mais le protocole QWERTY reste plus proche des données d’entraînement.

## Confidentialité

La dynamique de frappe est une donnée comportementale sensible.

Le prototype doit donc privilégier :

- la capture des événements nécessaires au calcul des features ;
- l’absence de diagnostic médical ;
- l’absence de stockage inutile du texte tapé ;
- l’utilisation locale ou contrôlée des résultats.

## Limites

Les limites principales sont :

- le modèle est entraîné sur NeuroQWERTY, pas sur des données navigateur modernes ;
- le nombre de sujets d’entraînement reste limité ;
- les conditions de saisie peuvent varier selon le clavier, le navigateur et l’utilisateur ;
- la modalité clavier est insuffisante comme test autonome ;
- le score doit être interprété comme un signal expérimental destiné à la fusion multimodale.
