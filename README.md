# kitchen

Manipulate counts matrix files and cook scRNA-seq data from command line

[![Latest Version][tag-version]][repo-url]

## Installing the kitchen

You can install a local version of the package (along with python dependencies) by cloning this repository and running the following command from the main directory:

```bash
pip install -e .
```

## Cooking in the kitchen

This package is intended to automate manipulation and processing of scRNA-seq counts matrices from the command line, eliminating the need for interactive python sessions with repetitive, manual `scanpy` functions.

```bash
# print out all kitchen command options
kitchen -h

# ex1: print information about anndata object (saved in .h5ad format) to console
kitchen info <path/to/.h5ad>

# ex2: process a filtered .h5ad file from raw counts, performing unsupervised clustering,
# cell cycle inference, and UMAP embedding colored by genes, mito percentage,
# and cell cycle phase, along with leiden clusters and PAGA graph
kitchen recipe <path/to/.h5ad> -p -cc -c arcsinh_n_genes_by_counts pct_counts_mito phase
```

Full documentation is available at [codyheiser.github.io/kitchen/](https://codyheiser.github.io/kitchen/).

[tag-version]: https://img.shields.io/github/v/tag/codyheiser/kitchen
[repo-url]: https://github.com/codyheiser/kitchen
