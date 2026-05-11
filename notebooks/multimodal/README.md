# Notebooks multimodaux

Ce dossier contient les expérimentations de fusion tardive entre les modalités clavier, voix et dessin.

## Notebook principal

```text
multimodal_fusion_analysis.ipynb
```

Ce notebook lit les scores out-of-fold générés par :

```bash
python scripts/generate_unimodal_oof_scores.py
```

Il doit être lu comme une analyse exploratoire cross-dataset. Les sujets ne sont pas réellement multimodaux : les échantillons sont appariés par label uniquement.

La méthodologie associée est décrite dans :

```text
docs/multimodal/evaluation_pseudo_cohorte.md
```
