# Domain adaptive neural networks for population genetic inference

This repository contains code to implement the domain adaptation framework for population genetic inference using TensorFlow as proposed in:

> Mo Z, Siepel A. 2023. Domain-adaptive neural networks improve supervised machine learning based on simulated population genetic data. bioRxiv:2023.03.01.529396. [doi.org/10.1101/2023.03.01.529396](https://doi.org/10.1101/2023.03.01.529396)


### 1. **D**omain **ada**ptive SIA (dadaSIA)

The `DA-SIA` directory contains building blocks of the dadaSIA model, a domain adaptive version of the [original SIA model](https://doi.org/10.1093/molbev/msab332).

The `encode()` function in `fea_encoding.py` is an improved genealogical encoding used in dadaSIA (see **Fig. S1B** in the manuscript for details). The function takes the following input:
```python
encode(nwk_str, no_taxa, der_ls)
```

| Input | Data type | Description |
| ----- | --------- | ----------- |
| `nwk_str` | string | Newick encoding of the genealogy |
| `no_taxa` | int | Number of taxa in the genealogy |
| `der_ls` | list of strings, or None | Taxon labels of the derived alleles |

`encode()` returns a tuple of 3 matrices (as numpy arrays) `(F, W, R)`, or 2 matrices `(F, W)` if `der_ls` is None.

The dadaSIA neural network models for sweep classification and selection coefficient inference are defined in `SIA_GRL.py` and `SIA_sc_GRL.py`, respectively. In addition, an example of a custom data generator for training the domain-adaptive model is provided in each script.


### 2. Domain adaptive ReLERNN

`ReLERNN_mods.py` in the `DA-ReLERNN` directory implements a domain-adaptive neural network modeled after ReLERRN, originally introduced by [Adrion et al.](https://doi.org/10.1093/molbev/msaa038). An example of a data generator for training the model is provided in the script.
