# -*- coding: utf-8 -*-
"""
Functions for manipulating .h5ad objects and automated processing of scRNA-seq data
"""
import os, errno
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import ceil
from matplotlib import rcParams
from emptydrops import find_nonambient_barcodes
from emptydrops.matrix import CountMatrix

from .plotting import custom_heatmap

sc.set_figure_params(frameon=False, dpi=200, dpi_save=300, format="png")

# define cell cycle phase genes
#  human genes ('_h') from satija lab list
#  mouse genes ('_m') converted from human list using biomaRt package in R
s_genes_h = [
    "MCM5",
    "PCNA",
    "TYMS",
    "FEN1",
    "MCM2",
    "MCM4",
    "RRM1",
    "UNG",
    "GINS2",
    "MCM6",
    "CDCA7",
    "DTL",
    "PRIM1",
    "UHRF1",
    "MLF1IP",
    "HELLS",
    "RFC2",
    "RPA2",
    "NASP",
    "RAD51AP1",
    "GMNN",
    "WDR76",
    "SLBP",
    "CCNE2",
    "UBR7",
    "POLD3",
    "MSH2",
    "ATAD2",
    "RAD51",
    "RRM2",
    "CDC45",
    "CDC6",
    "EXO1",
    "TIPIN",
    "DSCC1",
    "BLM",
    "CASP8AP2",
    "USP1",
    "CLSPN",
    "POLA1",
    "CHAF1B",
    "BRIP1",
    "E2F8",
]
s_genes_m = [
    "Gmnn",
    "Rad51",
    "Cdca7",
    "Pold3",
    "Slbp",
    "Prim1",
    "Dscc1",
    "Rad51ap1",
    "Fen1",
    "Mcm4",
    "Ccne2",
    "Tyms",
    "Rrm2",
    "Usp1",
    "Wdr76",
    "Mcm2",
    "Ung",
    "E2f8",
    "Exo1",
    "Chaf1b",
    "Blm",
    "Clspn",
    "Cdc45",
    "Cdc6",
    "Hells",
    "Nasp",
    "Ubr7",
    "Casp8ap2",
    "Mcm5",
    "Uhrf1",
    "Pcna",
    "Rrm1",
    "Rfc2",
    "Tipin",
    "Brip1",
    "Gins2",
    "Dtl",
    "Pola1",
    "Rpa2",
    "Mcm6",
    "Msh2",
]
g2m_genes_h = [
    "HMGB2",
    "CDK1",
    "NUSAP1",
    "UBE2C",
    "BIRC5",
    "TPX2",
    "TOP2A",
    "NDC80",
    "CKS2",
    "NUF2",
    "CKS1B",
    "MKI67",
    "TMPO",
    "CENPF",
    "TACC3",
    "FAM64A",
    "SMC4",
    "CCNB2",
    "CKAP2L",
    "CKAP2",
    "AURKB",
    "BUB1",
    "KIF11",
    "ANP32E",
    "TUBB4B",
    "GTSE1",
    "KIF20B",
    "HJURP",
    "CDCA3",
    "HN1",
    "CDC20",
    "TTK",
    "CDC25C",
    "KIF2C",
    "RANGAP1",
    "NCAPD2",
    "DLGAP5",
    "CDCA2",
    "CDCA8",
    "ECT2",
    "KIF23",
    "HMMR",
    "AURKA",
    "PSRC1",
    "ANLN",
    "LBR",
    "CKAP5",
    "CENPE",
    "CTCF",
    "NEK2",
    "G2E3",
    "GAS2L3",
    "CBX5",
    "CENPA",
]
g2m_genes_m = [
    "Tmpo",
    "Smc4",
    "Tacc3",
    "Cdk1",
    "Ckap2l",
    "Cks2",
    "Mki67",
    "Ckap5",
    "Nusap1",
    "Top2a",
    "Ect2",
    "Cdca3",
    "Cdc25c",
    "Cks1b",
    "Kif11",
    "Psrc1",
    "Dlgap5",
    "Ckap2",
    "Aurkb",
    "Ttk",
    "Cdca8",
    "Cdca2",
    "Cdc20",
    "Lbr",
    "Birc5",
    "Kif20b",
    "Nuf2",
    "Anp32e",
    "Cenpf",
    "Kif2c",
    "Ube2c",
    "Cenpa",
    "Rangap1",
    "Hjurp",
    "Ndc80",
    "Ncapd2",
    "Anln",
    "Cenpe",
    "Cbx5",
    "Hmgb2",
    "Gas2l3",
    "Cks1brt",
    "Bub1",
    "Gtse1",
    "Nek2",
    "G2e3",
    "Tpx2",
    "Hmmr",
    "Aurka",
    "Ccnb2",
    "Ctcf",
    "Tubb4b",
    "Kif23",
]


def check_dir_exists(path):
    """
    Checks if directory already exists or not and creates it if it doesn't

    Parameters
    ----------

    path : str
        path to directory

    Returns
    -------

    tries to make directory at `path`, unless it already exists
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def list_union(lst1, lst2):
    """
    Combines two lists by the union of their values

    Parameters
    ----------

    lst1, lst2 : list
        lists to combine

    Returns
    -------

    final_list : list
        union of values in lst1 and lst2
    """
    final_list = set(lst1).union(set(lst2))
    return final_list


def cellranger2(
    adata,
    expected=1500,
    upper_quant=0.99,
    lower_prop=0.1,
    label="CellRanger_2",
    verbose=True,
):
    """
    Labels cells using "knee point" method from CellRanger 2.1

    Parameters
    ----------

    adata : anndata.AnnData
        object containing unfiltered counts
    expected : int, optional (default=1500)
        estimated number of real cells expected in dataset
    upper_quant : float, optional (default=0.99)
        upper quantile of real cells to test
    lower_prop : float, optional (default=0.1)
        percentage of expected quantile to calculate total counts threshold for
    label : str, optional (default="CellRanger_2")
        how to name .obs column containing output
    verbose : bool, optional (default=True)
        print updates to console

    Returns
    -------

    adata edited in place to add .obs[label] binary label
    """
    if "total_counts" not in adata.obs.columns:
        adata.obs["total_counts"] = adata.X.sum(axis=1)
    tmp = np.sort(np.array(adata.obs["total_counts"]))[::-1]
    thresh = np.quantile(tmp[0:expected], upper_quant) * lower_prop
    adata.uns["{}_knee_thresh".format(label)] = thresh
    adata.obs[label] = "False"
    adata.obs.loc[adata.obs.total_counts > thresh, label] = "True"
    adata.obs[label] = adata.obs[label].astype("category")
    if verbose:
        print("Detected knee point: {}".format(round(thresh, 3)))
        print(adata.obs[label].value_counts())


def cellranger3(
    adata,
    init_counts=15000,
    min_umi_frac_of_median=0.01,
    min_umis_nonambient=500,
    max_adj_pvalue=0.01,
):
    """
    Labels cells using "emptydrops" method from CellRanger 3.0

    Parameters
    ----------

    adata : anndata.AnnData
        object containing unfiltered counts
    init_counts : int, optional (default=15000)
        initial total counts threshold for calling cells
    min_umi_frac_of_median : float, optional (default=0.01)
        minimum total counts for testing barcodes as fraction of median counts for
        initially labeled cells
    min_umis_nonambient : float, optional (default=500)
        minimum total counts for testing barcodes
    max_adj_pvalue : float, optional (default=0.01)
        maximum p-value for cell calling after B-H correction

    Returns
    -------

    adata edited in place to add .obs["CellRanger_3"] binary label
    and .obs["CellRanger_3_ll"] log-likelihoods for tested barcodes
    """
    m = CountMatrix.from_anndata(adata)  # create emptydrops object from adata
    if "total_counts" not in adata.obs.columns:
        adata.obs["total_counts"] = adata.X.sum(axis=1)
    # label initial cell calls above total counts threshold
    adata.obs["CellRanger_3"] = False
    adata.obs.loc[adata.obs.total_counts > init_counts, "CellRanger_3"] = True
    # call emptydrops to test for nonambient barcodes
    out = find_nonambient_barcodes(
        m,
        np.array(
            adata.obs.loc[adata.obs.CellRanger_3 == True,].index,
            dtype=m.bcs.dtype,
        ),
        min_umi_frac_of_median=min_umi_frac_of_median,
        min_umis_nonambient=min_umis_nonambient,
        max_adj_pvalue=max_adj_pvalue,
    )
    # assign binary labels from emptydrops
    adata.obs.CellRanger_3.iloc[out[0]] = out[-1]
    adata.obs.CellRanger_3 = adata.obs.CellRanger_3.astype(str).astype(
        "category"
    )  # convert to category
    # assign log-likelihoods from emptydrops to .obs
    adata.obs["CellRanger_3_ll"] = 0
    adata.obs.CellRanger_3_ll.iloc[out[0]] = out[1]


def subset_adata(adata, subset, verbose=True):
    """
    Subsets AnnData object on one or more .obs columns

    Columns should contain 0/False for cells to throw out, and 1/True for cells to
    keep. Keeps union of all labels provided in subset.

    Parameters
    ----------

    adata : anndata.AnnData
        the data
    subset : str or list of str
        adata.obs labels to use for subsetting. Labels must be binary (0, "0", False,
        "False" to toss - 1, "1", True, "True" to keep). Multiple labels will keep
        intersection.
    verbose : bool, optional (default=True)
        print updates to console

    Returns
    -------

    adata : anndata.AnnData
        new anndata object as subset of `adata`
    """
    if verbose:
        print("Subsetting AnnData on {}".format(subset), end="")
    if isinstance(subset, str):
        subset = [subset]
    # initialize .obs column for choosing cells
    adata.obs["adata_subset_combined"] = 0
    # create label as union of given subset args
    for i in range(len(subset)):
        adata.obs.loc[
            adata.obs[subset[i]].isin(["True", True, 1.0, 1]), "adata_subset_combined"
        ] = 1
    adata = adata[adata.obs["adata_subset_combined"] == 1, :].copy()
    adata.obs.drop(columns="adata_subset_combined", inplace=True)
    if verbose:
        print(" - now {} cells and {} genes".format(adata.n_obs, adata.n_vars))
    return adata


def cc_score(adata, layer=None, seed=18, verbose=True):
    """
    Calculates cell cycle scores and implied phase for each observation

    Parameters
    ----------

    adata : anndata.AnnData
        object containing transformed and normalized (arcsinh or log1p) counts in
        'layer'.
    layer : str, optional (default=None)
        key from adata.layers to use for cc phase calculation. Default None to
        use .X
    seed : int, optional (default=18)
        random state for PCA, neighbors graph and clustering
    verbose : bool, optional (default=True)
        print updates to console

    Returns
    -------

    adata is edited in place to add 'G2M_score', 'S_score', and 'phase' to .obs
    """
    if layer is not None:
        adata.layers["temp"] = adata.X.copy()
        adata.X = adata.layers[layer].copy()
        if verbose:
            print("Calculating cell cycle scores using layer: {}".format(layer))
    else:
        if verbose:
            print("Calculating cell cycle scores")
    # determine if sample is mouse or human based on gene names
    if any(item in adata.var_names for item in s_genes_h + g2m_genes_h):
        s_genes, g2m_genes = s_genes_h, g2m_genes_h
    elif any(item in adata.var_names for item in s_genes_m + g2m_genes_m):
        s_genes, g2m_genes = s_genes_m, g2m_genes_m
    # score cell cycle using scanpy function
    sc.tl.score_genes_cell_cycle(
        adata,
        s_genes=s_genes,  # defined at top of script
        g2m_genes=g2m_genes,  # defined at top of script
        random_state=seed,
    )
    if layer is not None:
        adata.X = adata.layers["temp"].copy()
        del adata.layers["temp"]


def dim_reduce(
    adata,
    layer=None,
    use_rep=None,
    clust_resolution=1.0,
    paga=True,
    seed=18,
    verbose=True,
):
    """
    Reduces dimensions of single-cell dataset using standard methods

    Parameters
    ----------

    adata : anndata.AnnData
        object containing preprocessed counts matrix
    layer : str, optional (default=None)
        layer to use; default None for .X
    use_rep : str, optional (default=None)
        .obsm key to use for neighbors graph instead of PCA;
        default None, generate new PCA from layer
    clust_resolution : float, optional (default=1.0)
        resolution as fraction on [0.0, 1.0] for leiden
        clustering. default 1.0
    paga : bool, optional (default=True)
        run PAGA to seed UMAP embedding
    seed : int, optional (default=18)
        random state for PCA, neighbors graph and clustering
    verbose : bool, optional (default=True)
        print updates to console

    Returns
    -------

    adata is edited in place, adding PCA, neighbors graph, PAGA, and UMAP
    """
    if use_rep is None:
        if layer is not None:
            if verbose:
                print("Using layer {} for dimension reduction".format(layer))
            adata.X = adata.layers[layer].copy()
        if verbose:
            print(
                "Performing {}-component PCA and building kNN graph with {} neighbors".format(
                    "50" if adata.n_obs >= 50 else adata.n_obs,
                    int(np.sqrt(adata.n_obs)),
                )
            )
        sc.pp.pca(
            adata,
            n_comps=50 if adata.n_obs >= 50 else adata.n_obs - 1,
            random_state=seed,
        )
        sc.pp.neighbors(
            adata,
            n_neighbors=int(np.sqrt(adata.n_obs)),
            n_pcs=20 if adata.n_obs >= 50 else int(0.4 * (adata.n_obs - 1)),
            random_state=seed,
        )
    else:
        if verbose:
            print(
                "Building kNN graph with {} nearest neighbors using {}".format(
                    int(np.sqrt(adata.n_obs)), use_rep
                )
            )
        sc.pp.neighbors(
            adata,
            n_neighbors=int(np.sqrt(adata.n_obs)),
            use_rep=use_rep,
            random_state=seed,
        )
    if verbose:
        print("Clustering cells using Leiden algorithm ", end="")
    sc.tl.leiden(adata, random_state=seed, resolution=clust_resolution)
    if verbose:
        print(
            "- {} clusters identified".format(
                (len(adata.obs.leiden.cat.categories) + 1)
            )
        )
    if paga:
        if verbose:
            print("Building PAGA graph and UMAP from coordinates")
        sc.tl.paga(adata)
        sc.pl.paga(adata, show=False)
        sc.tl.umap(adata, init_pos="paga", random_state=seed)
    else:
        sc.tl.umap(adata, random_state=seed)


def plot_genes(
    adata,
    de_method="t-test_overestim_var",
    layer="log1p_norm",
    groupby="leiden",
    ambient=False,
    plot_type=None,
    n_genes=5,
    dendrogram=True,
    cmap="Greys",
    figsize_scale=1.0,
    save_to="de.png",
    verbose=True,
    **kwargs,
):
    """
    Calculates and plot `rank_genes_groups` results

    Parameters
    ----------
    adata : anndata.AnnData
        object containing preprocessed and dimension-reduced counts matrix
    de_method : str, optional (default="t-test_overestim_var")
        one of "t-test", "t-test_overestim_var", "wilcoxon"
    layer : str, optional (default="log1p_norm")
        one of `adata.layers` to use for DEG analysis. Recommended to use 'raw_counts'
        if `de_method=='wilcoxon'` and 'log1p_norm' if `de_method=='t-test'`.
    groupby : str, optional (default="leiden")
        `adata.obs` key to group cells by
    ambient : bool, optional (default=False)
        include ambient genes as a group in the plot/output dictionary
    plot_type : str, optional (default=`None`)
        One of "dotplot", "matrixplot", "dotmatrix", "stacked_violin", or "heatmap".
        If `None`, don't plot, just return DEGs as dictionary.
    n_genes : int, optional (default=5)
        number of top genes per group to show
    dendrogram : bool, optional (default=True)
        show dendrogram of cluster similarity
    cmap : str, optional (default="Greys")
        matplotlib colormap for dots
    figsize_scale : float, optional (default=1.0)
        scale dimensions of the figure
    save_to : str, optional (default="de.png")
        string to add to plot name using scanpy plot defaults
    verbose : bool, optional (default=True)
        print updates to console
    **kwargs : optional
        keyword args to add to `kitchen.plotting.custom_heatmap`

    Returns
    -------
    markers : dict
        dictionary of top `n_genes` DEGs per group
    myplot : matplotlib.Figure
        `custom_heatmap` object if `plot_type!=None`
    """
    if verbose:
        print("Performing differential expression analysis...")
    assert de_method in [
        "t-test",
        "t-test_overestim_var",
        "wilcoxon",
    ], "Invalid de_method. Must be one of ['t-test','t-test_overestim_var','wilcoxon']."
    sc.tl.rank_genes_groups(
        adata, groupby=groupby, layer=layer, use_raw=False, method=de_method
    )

    # unique groups in DEG analysis
    groups = adata.obs[groupby].unique().tolist()

    # get markers manually
    markers = {}
    for clu in groups:
        markers[clu] = [
            adata.uns["rank_genes_groups"]["names"][x][clu] for x in range(n_genes)
        ]
    # append ambient genes
    if ambient:
        markers["ambient"] = adata.var_names[adata.var.ambient].tolist()

    # total and unique features on plot
    features = [item for sublist in markers.values() for item in sublist]
    unique_features = list(set(features))

    print(
        "Detected {} total genes and {} unique genes across {} groups".format(
            len(features), len(unique_features), len(groups)
        )
    )

    # plotting workflow if desired
    if plot_type is not None:
        # plot dimensions (long vertical)
        if plot_type in ["dotplot", "matrixplot", "stacked_violin"]:
            figsize = (
                len(groups) / 2 * figsize_scale,
                len(features) / 10 * figsize_scale,
            )
        # plot dimensions (long horizontal)
        elif plot_type == "heatmap":
            figsize = (
                len(features) / 10 * figsize_scale,
                len(groups) / 2 * figsize_scale,
            )

        # build plot
        my_plot = custom_heatmap(
            adata,
            groupby=groupby,
            features=unique_features,
            layer=layer,
            vars_dict=markers,
            cluster_obs=dendrogram,
            plot_type=plot_type,
            cmap=cmap,
            figsize=figsize,
            save=save_to,
            **kwargs,
        )
        return (markers, my_plot)

    else:
        return markers


def plot_genes_cnmf(
    adata,
    plot_type="heatmap",
    groupby="leiden",
    attr="varm",
    keys="cnmf_spectra",
    indices=None,
    n_genes=5,
    dendrogram=True,
    figsize_scale=1.0,
    cmap="Greys",
    save_to="de_cnmf.png",
    **kwargs,
):
    """
    Calculates and plots top cNMF gene loadings

    Parameters
    ----------
    adata : anndata.AnnData
        object containing preprocessed and dimension-reduced counts matrix
    plot_type : str, optional (default=`None`)
        One of "dotplot", "matrixplot", "dotmatrix", "stacked_violin", or "heatmap".
    groupby : str, optional (default="leiden")
        .obs key to group cells by
    attr : str {"var", "obs", "uns", "varm", "obsm"}
        attribute of adata that contains the score
    keys : str or list of str, optional (default="cnmf_spectra")
        scores to look up an array from the attribute of adata
    indices : list of int, optional (default=None)
        column indices of keys for which to plot (e.g. [0,1,2] for first three keys)
    n_genes : int, optional (default=5)
        number of top genes per group to show
    dendrogram : bool, optional (default=True)
        show dendrogram of cluster similarity
    figsize_scale : float, optional (default=1.0)
        scale dimensions of the figure
    cmap : str, optional (default="Greys")
        valid color map for the plot
    save_to : str, optional (default="de.png")
        string to add to plot name using scanpy plot defaults
    **kwargs : optional
        keyword args to add to sc.pl.matrixplot, sc.pl.dotplot, or sc.pl.heatmap

    Returns
    -------
    markers : dict
        dictionary of top `n_genes` features per NMF factor
    myplot : matplotlib.Figure
        `custom_heatmap` object
    """
    # calculate arcsinh counts for visualization
    adata.X = adata.layers["raw_counts"].copy()
    sc.pp.normalize_total(adata)
    adata.X = np.arcsinh(adata.X)
    adata.layers["arcsinh"] = adata.X.copy()
    adata.X = adata.layers["raw_counts"].copy()  # return raw counts to .X

    # default to all usages
    if indices is None:
        indices = [x for x in range(getattr(adata, attr)[keys].shape[1])]
    # get scores for each usage
    if isinstance(keys, str) and indices is not None:
        scores = np.array(getattr(adata, attr)[keys])[:, indices]
        keys = ["{}_{}".format(keys, i + 1) for i in indices]
    labels = adata.var_names  # search all var_names for top genes based on spectra
    # get top n_genes for each spectra
    markers = {}
    for iscore, score in enumerate(scores.T):
        markers[keys[iscore]] = []
        indices = np.argsort(score)[::-1][:n_genes]
        for x in indices[::-1]:
            markers[keys[iscore]].append(labels[x])

    # total and unique features on plot
    features = [item for sublist in markers.values() for item in sublist]
    unique_features = list(set(features))

    # unique groups in DEG analysis
    groups = adata.obs[groupby].unique().tolist()

    # plot dimensions (long vertical)
    if plot_type in ["dotplot", "matrixplot", "stacked_violin"]:
        figsize = (len(groups) / 2 * figsize_scale, len(features) / 10 * figsize_scale)
    # plot dimensions (long horizontal)
    elif plot_type == "heatmap":
        figsize = (len(features) / 10 * figsize_scale, len(groups) / 2 * figsize_scale)

    # build plot
    my_plot = custom_heatmap(
        adata,
        groupby=groupby,
        features=unique_features,
        vars_dict=markers,
        cluster_obs=dendrogram,
        plot_type=plot_type,
        cmap=cmap,
        figsize=figsize,
        save=save_to,
        **kwargs,
    )
    return (markers, my_plot)


def rank_genes_cnmf(
    adata,
    attr="varm",
    keys="cnmf_spectra",
    indices=None,
    labels=None,
    titles=None,
    color="black",
    n_points=20,
    ncols=5,
    log=False,
    show=None,
    figsize=(5, 5),
):
    """
    Plots rankings. [Adapted from `scanpy.plotting._anndata.ranking`]

    See, for example, how this is used in `pl.pca_ranking`.

    Parameters
    ----------

    adata : anndata.AnnData
        the data
    attr : str {'var', 'obs', 'uns', 'varm', 'obsm'}
        the attribute of adata that contains the score
    keys : str or list of str, optional (default="cnmf_spectra")
        scores to look up an array from the attribute of adata
    indices : list of int, optional (default=None)
        the column indices of keys for which to plot (e.g. [0,1,2] for first three
        keys)
    labels : list of str, optional (default=None)
        Labels to use for features displayed as plt.txt objects on the axes
    titles : list of str, optional (default=None)
        Labels for titles of each plot panel, in order
    ncols : int, optional (default=5)
        number of columns in gridspec
    show : bool, optional (default=None)
        show figure or just return axes
    figsize : tuple of float, optional (default=(5,5))
        size of matplotlib figure

    Returns
    -------

    matplotlib gridspec with access to the axes
    """
    # default to all usages
    if indices is None:
        indices = [x for x in range(getattr(adata, attr)[keys].shape[1])]
    # get scores for each usage
    if isinstance(keys, str) and indices is not None:
        scores = np.array(getattr(adata, attr)[keys])[:, indices]
        keys = ["{}_{}".format(keys, i + 1) for i in indices]
    n_panels = len(indices) if isinstance(indices, list) else 1
    if n_panels == 1:
        scores, keys = scores[:, None], [keys]
    if log:
        scores = np.log(scores)
    if labels is None:
        labels = (
            adata.var_names
            if attr in {"var", "varm"}
            else np.arange(scores.shape[0]).astype(str)
        )
    if titles is not None:
        assert len(titles) == n_panels, "Must provide {} titles".format(n_panels)
    if isinstance(labels, str):
        labels = [labels + str(i + 1) for i in range(scores.shape[0])]
    if n_panels <= ncols:
        n_rows, n_cols = 1, n_panels
    else:
        n_rows, n_cols = ceil(n_panels / ncols), ncols
    fig = plt.figure(figsize=(n_cols * figsize[0], n_rows * figsize[1]))
    left, bottom = 0.1 / n_cols, 0.1 / n_rows
    gs = gridspec.GridSpec(
        nrows=n_rows,
        ncols=n_cols,
        wspace=0.1,
        left=left,
        bottom=bottom,
        right=1 - (n_cols - 1) * left - 0.01 / n_cols,
        top=1 - (n_rows - 1) * bottom - 0.1 / n_rows,
    )
    for iscore, score in enumerate(scores.T):
        plt.subplot(gs[iscore])
        indices = np.argsort(score)[::-1][: n_points + 1]
        for ig, g in enumerate(indices[::-1]):
            plt.text(
                x=score[g],
                y=ig,
                s=labels[g],
                color=color,
                verticalalignment="center",
                horizontalalignment="right",
                fontsize="medium",
                fontstyle="italic",
            )
        if titles is not None:
            plt.title(titles[iscore], fontsize="x-large")
        else:
            plt.title(keys[iscore].replace("_", " "), fontsize="x-large")
        plt.ylim(-0.9, ig + 0.9)
        score_min, score_max = np.min(score[indices]), np.max(score[indices])
        plt.xlim(
            (0.95 if score_min > 0 else 1.05) * score_min,
            (1.05 if score_max > 0 else 0.95) * score_max,
        )
        plt.xticks(rotation=45)
        plt.tick_params(labelsize="medium")
        plt.tick_params(
            axis="y",  # changes apply to the y-axis
            which="both",  # both major and minor ticks are affected
            left=False,
            right=False,
            labelleft=False,
        )
        plt.grid(False)
    gs.tight_layout(fig)
    if show == False:
        return gs
