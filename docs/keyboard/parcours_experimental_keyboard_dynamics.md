# Parcours expérimental - Modalité clavier

Ce document retrace les principales étapes suivies pour construire la modalité **Keyboard Dynamics** du projet. Il explique les choix effectués, les pistes testées, les résultats importants et la méthode finalement retenue.

L’objectif n’était pas seulement d’obtenir le meilleur score possible, mais de construire une méthode exploitable dans une application web, avec une validation suffisamment stricte pour éviter les résultats trop optimistes.

## 1. Point de départ

L’idée initiale était d’utiliser la dynamique de frappe comme indicateur moteur potentiel. Les signaux utilisés sont principalement :

- la durée d’appui sur les touches (`hold time`) ;
- le délai entre deux frappes (`flight time`) ;
- la vitesse de frappe ;
- la variabilité du rythme ;
- la stabilité du signal sur plusieurs segments.

Le travail s’est appuyé sur l’état de l’art autour de NeuroQWERTY et sur un papier récent utilisant PSO et des modèles d’ensemble. Les scores annoncés dans ce papier étaient très élevés, mais ils ont été considérés avec prudence, notamment à cause du risque de fuite de données si les splits ne sont pas groupés par sujet.

## 2. Choix du dataset

Deux datasets ont été étudiés.

### NeuroQWERTY

NeuroQWERTY a été retenu comme dataset principal parce qu’il est plus encadré et plus cohérent avec l’idée d’un test standardisé dans le navigateur.

Il contient :

- `85` sujets ;
- `116` sessions disponibles ;
- des logs bruts de frappe ;
- un label contrôle/Parkinson ;
- les datasets MIT-CS1PD et MIT-CS2PD.

### Tappy

Tappy a aussi été exploré. Il contient plus de données, mais son protocole est moins contrôlé. Les données viennent d’un usage plus naturel, avec plus de variabilité difficile à maîtriser.

Tappy a donc été utile pour comparer et tester des idées, mais il n’a pas été retenu pour le modèle final.

## 3. Premières baselines

Les premiers notebooks ont servi à comprendre les données et à construire une baseline :

- chargement des fichiers bruts ;
- nettoyage des touches non pertinentes ;
- extraction des `hold_time` et `flight_time` ;
- analyse exploratoire ;
- premiers modèles sklearn classiques.

Ces premières expériences ont montré que le signal existe, mais que les scores peuvent devenir rapidement trop optimistes si la validation n’est pas assez stricte.

## 4. Passage à une évaluation groupée par sujet

Un point méthodologique important est apparu rapidement : plusieurs sessions peuvent appartenir au même sujet. Il faut donc éviter qu’un même sujet soit présent à la fois dans le train et dans la validation.

La validation retenue utilise donc des splits groupés par `pID`.

Cette contrainte est essentielle :

```text
train subjects ∩ validation subjects = ∅
```

Sans cette séparation, le modèle risque d’apprendre des caractéristiques individuelles de frappe plutôt que des différences liées au label contrôle/Parkinson.

## 5. Segmentation des sessions

Pour rapprocher le modèle d’un futur test navigateur, les sessions ont été découpées en segments de frappe.

Le principe est :

1. nettoyer les frappes ;
2. découper la session en fenêtres ;
3. extraire des features par segment ;
4. prédire une probabilité par segment ;
5. agréger les probabilités au niveau session.

Cette approche est plus adaptée à l’application qu’un modèle entraîné directement sur des sessions complètes de longueur variable.

## 6. Tests de sélection de features avec PSO

Une piste issue de l’état de l’art était d’utiliser PSO pour sélectionner les features.

Les essais PSO ont montré que certaines variantes pouvaient améliorer les scores en exploration, notamment des variantes `timing-only`. Ces variantes sont intéressantes parce qu’elles évitent les features dépendantes du layout clavier.

Cependant, une évaluation plus stricte a montré que les scores initiaux étaient optimistes. La sélection de features doit elle-même être intégrée à la validation interne, sinon elle peut suradapter les folds utilisés pour l’évaluation.

Conclusion : PSO est une piste intéressante, mais elle n’a pas été retenue pour le modèle final.

## 7. Évaluation stricte avec folds internes et externes

Une validation plus rigoureuse a ensuite été mise en place.

Le principe est :

- folds externes : mesurer la performance finale ;
- folds internes : choisir les paramètres, le modèle, les features et le seuil ;
- groupes : `pID`, pour ne jamais mélanger les sujets.

Cette approche est plus réaliste, mais elle donne des scores plus bas que les notebooks exploratoires.

Le notebook de sélection stricte a montré que :

- le signal clavier existe ;
- les performances restent variables selon les sujets ;
- les choix automatiques de configuration peuvent changer selon les folds ;
- les résultats réalistes sont plus modestes que certains scores de l’état de l’art.

## 8. Tests XGBoost, ondelettes et AUC

Ensuite, plusieurs représentations plus riches ont été testées :

- features temporelles agrégées ;
- features FFT ;
- features ondelettes ;
- combinaison de features ;
- modèles `HistGB`, `SVC` et `XGBoost`.

Les résultats ont montré que `XGBoost` exploitait bien les features tabulaires. Les variantes ondelettes amélioraient certaines métriques, mais elles étaient moins stables selon les folds.

Une distinction importante est apparue :

- certaines variantes donnaient un meilleur F1 binaire ;
- d’autres donnaient un meilleur score continu via ROC-AUC et PR-AUC.

Pour une fusion multimodale, le score continu est particulièrement important, car il sera combiné avec les scores des autres modalités.

## 9. Choix du modèle final

La méthode retenue est :

```text
agg_timing_xgb
```

Elle utilise des features temporelles agrégées et un modèle `XGBoost`.

Ce choix a été retenu pour plusieurs raisons :

- bonne ROC-AUC ;
- bonne PR-AUC ;
- modèle plus simple que les variantes DWT/DTW ;
- seulement `34` features ;
- intégration directe dans Flask ;
- score continu adapté à la fusion tardive.

L’artefact final est :

```text
models/keyboard_dynamics_neuroqwerty_agg_timing_xgb.joblib
```

## 10. Résultats du modèle final

L’évaluation finale donne :

| Métrique | Moyenne | Écart-type |
|---|---:|---:|
| Accuracy | 0.724 | 0.061 |
| Balanced accuracy | 0.715 | 0.074 |
| F1 macro | 0.706 | 0.080 |
| F1 Parkinson | 0.712 | 0.139 |
| Précision Parkinson | 0.762 | 0.095 |
| Rappel Parkinson | 0.715 | 0.226 |
| ROC-AUC | 0.834 | 0.039 |
| PR-AUC | 0.882 | 0.055 |

Matrice de confusion agrégée :

| Classe réelle | Prédit contrôle | Prédit Parkinson |
|---|---:|---:|
| Contrôle | 40 | 15 |
| Parkinson | 17 | 43 |

Ces résultats montrent que la modalité clavier est utile, mais pas suffisante seule. Le rappel Parkinson reste imparfait, ce qui justifie l’approche multimodale.

## 11. Dernière vérification : DWT/DTW + XGBoost

Une dernière expérience a testé une variante plus complexe combinant DWT, DTW et XGBoost.

Le but était de vérifier si une représentation temporelle plus riche pouvait battre le modèle final.

Résultats principaux :

| Variante | F1 macro | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| `agg_timing_xgb` | 0.767 ± 0.074 | 0.821 ± 0.060 | 0.879 ± 0.048 |
| `agg_dwt_dtw_xgb` | 0.739 ± 0.099 | 0.820 ± 0.058 | 0.881 ± 0.045 |
| `dwt_xgb` | 0.714 ± 0.111 | 0.820 ± 0.099 | 0.882 ± 0.065 |

La variante DWT/DTW ne donne pas de gain robuste. Elle ajoute de la complexité sans améliorer clairement les scores. Le modèle `agg_timing_xgb` reste donc le meilleur compromis.

## 12. Intégration dans l’application

Le prédicteur clavier utilise directement le modèle final :

```text
src/modalities/keyboard/predictor.py
```

Le pipeline applicatif est :

1. capture des événements clavier dans le navigateur ;
2. reconstruction des frappes valides ;
3. extraction des features `agg_timing_xgb` ;
4. prédiction par le pipeline `joblib` ;
5. moyenne des scores de segments ;
6. retour d’un `PredictionResult` standard ;
7. stockage du score pour la fusion multimodale.

## 13. Notebooks associés

Les notebooks principaux sont :

| Notebook | Rôle |
|---|---|
| `10_keyboard_dynamics_neuroqwerty_final_agg_timing_xgb.ipynb` | Modèle final et export. |
| `11_keyboard_dynamics_neuroqwerty_dwt_dtw_repeated_auc.ipynb` | Vérification DWT/DTW. |
| `06_keyboard_dynamics_neuroqwerty_rigorous_selection.ipynb` | Méthodologie stricte de sélection. |

Les autres notebooks documentent les explorations intermédiaires : EDA, Tappy, deep learning séquentiel, PSO, ondelettes et validation imbriquée.

## 14. Limites

Les principales limites sont :

- nombre limité de sujets ;
- absence de validation clinique externe ;
- protocole navigateur différent du protocole NeuroQWERTY original ;
- modèle clavier insuffisant comme test autonome ;
- sensibilité possible au clavier utilisé et aux conditions de saisie ;
- usage expérimental uniquement.

## 15. Conclusion

La modalité clavier fournit un score exploitable pour un prototype multimodal. Le modèle final `agg_timing_xgb` est un compromis raisonnable entre performance, stabilité, simplicité et intégration dans l’application web.

Les résultats ne permettent pas de conclure à une performance clinique, mais ils sont suffisants pour justifier l’utilisation du clavier comme une modalité complémentaire dans une fusion tardive avec d’autres signaux.
