# Notebooks

Les notebooks documentent les expérimentations réalisées pour la modalité clavier, depuis l’exploration initiale jusqu’au modèle retenu.

## Notebooks principaux

```text
10_keyboard_dynamics_neuroqwerty_final_agg_timing_xgb.ipynb
11_keyboard_dynamics_neuroqwerty_dwt_dtw_repeated_auc.ipynb
```

Le notebook `10` entraîne, évalue et exporte le modèle clavier utilisé dans l’application.

Le notebook `11` vérifie qu’une variante plus complexe `DWT/DTW + XGBoost` ne justifie pas de remplacer le modèle final.

## Méthodologie

```text
06_keyboard_dynamics_neuroqwerty_rigorous_selection.ipynb
```

Ce notebook documente la logique d’évaluation stricte avec folds externes et internes groupés par sujet.

## Notebooks exploratoires

Ces notebooks décrivent les essais intermédiaires et les pistes non retenues :

```text
01_keyboard_dynamics_neuroqwerty.ipynb
02_keyboard_dynamics_tappy.ipynb
03_keyboard_dynamics_tappy_sequence_dl.ipynb
04_keyboard_dynamics_neuroqwerty_v2_segments.ipynb
05_keyboard_dynamics_neuroqwerty_v2_pso.ipynb
07_keyboard_dynamics_neuroqwerty_notebook05_strict_eval.ipynb
08_keyboard_dynamics_neuroqwerty_auc_wavelet_xgboost.ipynb
09_keyboard_dynamics_neuroqwerty_xgboost_wavelet_nested.ipynb
```
