# -*- coding: utf-8 -*-
"""
Functions for manipulating .h5ad objects and automated processing of scRNA-seq data

@author: C Heiser
"""
import os, errno, argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import ceil
from emptydrops import find_nonambient_barcodes
from emptydrops.matrix import CountMatrix

sc.set_figure_params(
    color_map="viridis", frameon=False, dpi=100, dpi_save=200, format="png"
)

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
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def list_union(lst1, lst2):
    """
    Combine two lists by the union of their values
    """
    final_list = set(lst1).union(set(lst2))
    return final_list


def cellranger2(adata, expected=1500, upper_quant=0.99, lower_prop=0.1, verbose=True):
    """
    Label cells using "knee point" method from CellRanger 2.1

    Parameters:
        adata (anndata.AnnData): object containing unfiltered counts
        expected (int): estimated number of real cells expected in dataset
        upper_quant (float): upper quantile of real cells to test
        lower_prop (float): percentage of expected quantile to calculate 
            total counts threshold for
        verbose (bool): print updates to console

    Returns:
        adata edited in place to add .obs["CellRanger_2"] binary label
    """
    if "total_counts" not in adata.obs.columns:
        adata.obs["total_counts"] = adata.X.sum(axis=1)
    tmp = np.sort(np.array(adata.obs["total_counts"]))[::-1]
    thresh = np.quantile(tmp[0:expected], upper_quant) * lower_prop
    adata.uns["knee_thresh"] = thresh
    adata.obs["CellRanger_2"] = "False"
    adata.obs.loc[adata.obs.total_counts > thresh, "CellRanger_2"] = "True"
    adata.obs["CellRanger_2"] = adata.obs["CellRanger_2"].astype("category")
    if verbose:
        print("Detected knee point: {}".format(round(thresh, 3)))
        print(adata.obs.CellRanger_2.value_counts())


def cellranger3(
    adata,
    init_counts=15000,
    min_umi_frac_of_median=0.01,
    min_umis_nonambient=500,
    max_adj_pvalue=0.01,
):
    """
    Label cells using "emptydrops" method from CellRanger 3.0

    Parameters:
        adata (anndata.AnnData): object containing unfiltered counts
        init_counts (int): initial total counts threshold for calling cells
        min_umi_frac_of_median (float): minimum total counts for testing barcodes as
            fraction of median counts for initially labeled cells
        min_umis_nonambient (float): minimum total counts for testing barcodes
        max_adj_pvalue (float): maximum p-value for cell calling after B-H correction

    Returns:
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
            adata.obs.loc[adata.obs.CellRanger_3 == True,].index, dtype=m.bcs.dtype
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
    Subset AnnData object on one or more .obs columns
    columns should contain 0/False for cells to throw out, and 1/True for cells to keep
    keeps union of all labels provided in subset
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
    Calculate cell cycle scores and implied phase for each observation

    Parameters:
        adata (anndata.AnnData): object containing transformed and normalized
            (arcsinh or log1p) counts in 'layer'.
        layer (str): key from adata.layers to use for cc phase calculation.
            default None to use .X
        seed (int): random state for PCA, neighbors graph and clustering
        verbose (bool): print updates to console

    Returns:
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
    Reduce dimensions of single-cell dataset using standard methods

    Parameters:
        adata (anndata.AnnData): object containing preprocessed counts matrix
        layer (str): layer to use; default .X
        use_rep (str): .obsm key to use for neighbors graph instead of PCA;
            default None, generate new PCA from layer
        clust_resolution (float): resolution as fraction on [0.0, 1.0] for leiden
            clustering. default 1.0
        paga (bool): run PAGA to seed UMAP embedding
        seed (int): random state for PCA, neighbors graph and clustering
        verbose (bool): print updates to console

    Returns:
        adata is edited in place, adding PCA, neighbors graph, PAGA, and UMAP
    """
    if use_rep is None:
        if layer is not None:
            if verbose:
                print("Using layer {} for dimension reduction".format(layer))
            adata.X = adata.layers[layer].copy()
        if verbose:
            print(
                "Performing PCA and building kNN graph with {} nearest neighbors".format(
                    int(np.sqrt(adata.n_obs))
                )
            )
        sc.pp.pca(adata, n_comps=50, random_state=seed)
        sc.pp.neighbors(
            adata, n_neighbors=int(np.sqrt(adata.n_obs)), n_pcs=20, random_state=seed
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
            n_pcs=0,
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


def plot_embedding(
    adata, colors=None, show_clustering=True, ncols=4, save_to=None, verbose=True
):
    """
    Plot reduced-dimension embeddings of single-cell dataset

    Parameters:
        adata (anndata.AnnData): object containing preprocessed counts matrix
        colors (list of str): colors to plot; can be genes or .obs columns
        show_clustering (bool): plot PAGA graph and leiden clusters on first two axes
        ncols (int): number of columns in gridspec
        save_to (str): path to .png file for saving figure; default is plt.show()
        verbose (bool): print updates to console

    Returns:
        plot of PAGA, UMAP with Leiden and n_genes overlay, plus additional metrics
        from "colors"
    """
    if "paga" in adata.uns:
        cluster_colors = ["paga", "leiden"]
    else:
        cluster_colors = ["leiden"]
    if colors is not None:
        # get full list of things to plot on UMAP
        if show_clustering:
            colors = cluster_colors + colors
            unique_colors = list_union(cluster_colors, colors)
        else:
            unique_colors = set(colors)
    else:
        # with no colors provided, plot PAGA graph and leiden clusters
        colors = cluster_colors
        unique_colors = set(colors)
    if verbose:
        print("Plotting embedding with overlays: {}".format(list(unique_colors)))
    # set up figure size based on number of plots
    n_plots = len(unique_colors)
    if n_plots <= ncols:
        n_rows, n_cols = 1, n_plots
    else:
        n_rows, n_cols = ceil(n_plots / ncols), ncols
    fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
    # arrange axes as subplots
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    # add plots to axes
    i = 0
    for color in colors:
        if color in unique_colors:
            ax = plt.subplot(gs[i])
            if color.lower() == "paga":
                sc.pl.paga(
                    adata,
                    ax=ax,
                    frameon=False,
                    show=False,
                    fontsize="x-large",
                    fontoutline=2.5,
                    node_size_scale=3,
                )
            else:
                if color in ["leiden", "louvain", "cluster", "group", "cell_type"]:
                    leg_loc, leg_fontsize, leg_fontoutline = "on data", "x-large", 2.5
                else:
                    leg_loc, leg_fontsize, leg_fontoutline = (
                        "right margin",
                        "medium",
                        None,
                    )
                sc.pl.umap(
                    adata,
                    color=color,
                    ax=ax,
                    frameon=False,
                    show=False,
                    legend_loc=leg_loc,
                    legend_fontsize=leg_fontsize,
                    legend_fontoutline=leg_fontoutline,
                    size=50,
                )
                # add top three gene loadings if cNMF
                if color.str.startswith("usage_"):
                    [
                        axes[0].text(
                            x=0.5,
                            y=0.96 - (0.06 * x),
                            s="" + adata.uns["cnmf_spectra"].loc[x, color],
                            fontsize=12,
                            color="k",
                        )
                        for x in range(3)
                    ]
            unique_colors.remove(color)
            i = i + 1
    fig.tight_layout()
    if save_to is not None:
        if verbose:
            print("Saving embeddings to {}".format(save_to))
        plt.savefig(save_to)
    else:
        plt.show()


def plot_genes(
    adata,
    plot_type=["heatmap"],
    groupby="leiden",
    n_genes=5,
    dendrogram=True,
    ambient=False,
    cmap="viridis",
    save_to="_de.png",
    verbose=True,
):
    """
    Calculate and plot rank_genes_groups results

    Parameters:
        adata (anndata.AnnData): object containing preprocessed counts matrix
        plot_type (str): one or a list of combination of "heatmap", "dotplot", "matrixplot"
        groupby (str): .obs key to group cells by. default 'leiden'.
        dendrogram (bool): show dendrogram of cluster similarity
        ambient (bool): include ambient genes as a group in the plot
        cmap (str): valid color map for the plot
        save_to (str): string to add to plot name using scanpy plot defaults
        verbose (bool): print updates to console
    """
    if verbose:
        print("Performing differential expression analysis...")
    # rank genes with t-test and B-H correction
    sc.tl.rank_genes_groups(adata, groupby=groupby, layer="log1p_norm", use_raw=False)

    # calculate arcsinh counts for visualization
    adata.X = adata.layers["raw_counts"].copy()
    sc.pp.normalize_total(adata)
    adata.X = np.arcsinh(adata.X)
    adata.layers["arcsinh"] = adata.X.copy()
    adata.X = adata.layers["raw_counts"].copy()  # return raw counts to .X

    if isinstance(plot_type, str):
        plot_type = [plot_type]

    if ambient:
        # get markers manually and append ambient genes
        markers = {}
        for clu in adata.obs[groupby].unique().tolist():
            markers[clu] = [
                adata.uns["rank_genes_groups"]["names"][x][clu] for x in range(n_genes)
            ]
        markers["ambient"] = adata.var_names[adata.var.ambient].tolist()

        if "heatmap" in plot_type:
            sc.pl.heatmap(
                adata,
                markers,
                dendrogram=dendrogram,
                groupby=groupby,
                show_gene_labels=True,
                layer="arcsinh",
                var_group_rotation=0,
                cmap=cmap,
                save=save_to,
                show=False,
            )
        if "dotplot" in plot_type:
            sc.pl.dotplot(
                adata,
                markers,
                dendrogram=dendrogram,
                groupby=groupby,
                layer="arcsinh",
                var_group_rotation=0,
                color_map=cmap,
                save=save_to,
                show=False,
            )
        if "matrixplot" in plot_type:
            sc.pl.matrixplot(
                adata,
                markers,
                dendrogram=dendrogram,
                groupby=groupby,
                layer="arcsinh",
                var_group_rotation=0,
                cmap=cmap,
                save=save_to,
                show=False,
            )

    else:
        if "heatmap" in plot_type:
            sc.pl.rank_genes_groups_heatmap(
                adata,
                dendrogram=dendrogram,
                groupby=groupby,
                n_genes=n_genes,
                show_gene_labels=True,
                layer="arcsinh",
                var_group_rotation=0,
                cmap=cmap,
                save=save_to,
                show=False,
            )
        if "dotplot" in plot_type:
            sc.pl.rank_genes_groups_dotplot(
                adata,
                dendrogram=dendrogram,
                groupby=groupby,
                n_genes=n_genes,
                layer="arcsinh",
                var_group_rotation=0,
                color_map=cmap,
                save=save_to,
                show=False,
            )
        if "matrixplot" in plot_type:
            sc.pl.rank_genes_groups_matrixplot(
                adata,
                dendrogram=dendrogram,
                groupby=groupby,
                n_genes=n_genes,
                layer="arcsinh",
                var_group_rotation=0,
                cmap=cmap,
                save=save_to,
                show=False,
            )


def plot_genes_cnmf(
    adata,
    plot_type=["heatmap"],
    groupby="leiden",
    attr="varm",
    keys="cnmf_spectra",
    indices=None,
    n_genes=5,
    dendrogram=True,
    cmap="viridis",
    save_to="_de_cnmf.png",
):
    """
    Calculate and plot top cNMF gene loadings

    Parameters:
        adata (anndata.AnnData): object containing preprocessed counts matrix
        plot_type (str): one or a list of combination of "heatmap", "dotplot", "matrixplot"
        groupby (str): .obs key to group cells by. default 'leiden'.
        attr {'var', 'obs', 'uns', 'varm', 'obsm'}:
            The attribute of AnnData that contains the score.
        keys (str or list of str):
            The scores to look up an array from the attribute of adata.
        indices (list of int):
            The column indices of keys for which to plot (e.g. [0,1,2] for first three keys)
        dendrogram (bool): show dendrogram of cluster similarity
        cmap (str): valid color map for the plot
        save_to (str): string to add to plot name using scanpy plot defaults
    """
    # calculate arcsinh counts for visualization
    adata.X = adata.layers["raw_counts"].copy()
    sc.pp.normalize_total(adata)
    adata.X = np.arcsinh(adata.X)
    adata.layers["arcsinh"] = adata.X.copy()
    adata.X = adata.layers["raw_counts"].copy()  # return raw counts to .X

    if isinstance(plot_type, str):
        plot_type = [plot_type]

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

    if "heatmap" in plot_type:
        sc.pl.heatmap(
            adata,
            markers,
            dendrogram=dendrogram,
            groupby=groupby,
            show_gene_labels=True,
            layer="arcsinh",
            var_group_rotation=45,
            cmap=cmap,
            save=save_to,
            show=False,
        )
    if "dotplot" in plot_type:
        sc.pl.dotplot(
            adata,
            markers,
            dendrogram=dendrogram,
            groupby=groupby,
            layer="arcsinh",
            var_group_rotation=45,
            color_map=cmap,
            save=save_to,
            show=False,
        )
    if "matrixplot" in plot_type:
        sc.pl.matrixplot(
            adata,
            markers,
            dendrogram=dendrogram,
            groupby=groupby,
            layer="arcsinh",
            var_group_rotation=45,
            cmap=cmap,
            save=save_to,
            show=False,
        )


def rank_genes_cnmf(
    adata,
    attr="varm",
    keys="cnmf_spectra",
    indices=None,
    labels=None,
    color="black",
    n_points=20,
    log=False,
    show=None,
    figsize=(5, 5),
):
    """
    Plot rankings. [Adapted from scanpy.plotting._anndata.ranking]
    See, for example, how this is used in pl.pca_ranking.

    Parameters:
        adata : AnnData
            The data.
        attr : {'var', 'obs', 'uns', 'varm', 'obsm'}
            The attribute of AnnData that contains the score.
        keys : str or list of str
            The scores to look up an array from the attribute of adata.
        indices : list of int
            The column indices of keys for which to plot (e.g. [0,1,2] for first three keys)

    Returns:
        matplotlib gridspec with access to the axes.
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
    if isinstance(labels, str):
        labels = [labels + str(i + 1) for i in range(scores.shape[0])]
    if n_panels <= 5:
        n_rows, n_cols = 1, n_panels
    else:
        n_rows, n_cols = ceil(n_panels / 4), 4
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
                # rotation="vertical",
                verticalalignment="center",
                horizontalalignment="right",
                fontsize="medium",
            )
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
