# tabnet-facies
Apply Transformer model for mining well-logs data

# Baseline results

## Task 1: Facies classification
- For random forest, xgboost, lightgbm baseline:  use `run_xgb_geotut.py` and `xgb_localpreprocess.py`
- For Deep network  based model: use `run_dnn_baseline_geotut.py`
- For transformer based model: use `run_tabnet_geotut.py` and `run_tabnet_localwelllogs.py`

## Task 2: Well-logs embedding

- Use `run_embedding_geotut.py` and `run_embedding_localpreprocess.py`

# TODO

- [ ] Add feature engineerings to the set of raw features
- [ ] Add sequence-based features
- [ ] Test with other approaches on Transformer-based for tabular/sequence data

    
# References

1. Arik, S. O., & Pfister, T. (2019).
TabNet: Attentive Interpretable Tabular Learning. arXiv preprint arXiv:1908.07442.

2. Gorishniy, Yury, et al.
"Revisiting deep learning models for tabular data." Advances in Neural Information Processing Systems 34 (2021).
