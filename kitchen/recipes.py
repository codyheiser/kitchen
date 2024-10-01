# -*- coding: utf-8 -*-
"""
Functions for processing and manipulating .h5ad objects and automated processing of
scRNA-seq data
"""
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
from emptydrops import find_nonambient_barcodes
from emptydrops.matrix import CountMatrix

from .ingredients import (
    s_genes_h,
    s_genes_m,
    g2m_genes_h,
    g2m_genes_m,
    signature_dict_values,
    signature_dict_from_rank_genes_groups,
)
from .plotting import custom_heatmap, decoupler_dotplot_facet


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
        print(" - now {} cells and {} features".format(adata.n_obs, adata.n_vars))
    return adata


def calculate_cell_proportions(obs_df, celltype_column, groupby, celltype_subset=None):
    """
    calculate proportions of each cell type in a column

    Parameters
    ----------
    obs_df : pd.DataFrame
        Dataframe containing single cells as rows and metadata columns
    celltype_column : str
        Column of `obs_df` containing cell type labels for proportion calculation
    groupby : str
        Column of `obs_df` to group by for proportions (i.e. sample or patient)
    celltype_subset : list of str, optional (default=`None`)
        List of cell types in `obs_df.celltype_column` to keep in output. If `None`,
        return proportions of all celltypes

    Returns
    -------
    props_df : pd.DataFrame
        Dataframe with values of `obs_df.groupby` as rows and values of
        `obs_df.celltype_column` (or `celltype_subset`) as columns
    """
    # get total cells per group
    totals = obs_df.groupby(groupby).count()[[celltype_column]]
    totals.rename(columns={celltype_column: "total_cells"}, inplace=True)
    # count values in celltype_column per group
    counts = obs_df.groupby([groupby, celltype_column]).size().unstack(fill_value=0)
    # get only celltypes of interest
    if celltype_subset is not None:
        assert isinstance(
            celltype_subset, list
        ), "please provide 'celltype_subset' as a list"
        assert len(set(celltype_subset).intersection(set(counts.columns))) == len(
            celltype_subset
        ), "all values in 'celltype_subset' must be present in obs_df[celltype_column]"
        counts = counts[celltype_subset].copy()
    else:
        celltype_subset = list(counts.columns)
    # add totals
    props_df = counts.merge(totals, left_index=True, right_index=True)
    # calculate proportions
    for celltype in celltype_subset:
        props_df[celltype] = props_df[celltype] / props_df["total_cells"]

    return props_df


def score_gene_signatures(adata, signatures_dict, sig_subset=None, layer=None):
    """
    Score gene signatures in AnnData using `sc.tl.score_genes`

    Parameters
    ----------
    adata : anndata.AnnData
        Object containing gene expression data for scoring
    signatures_dict : dict
        Dictionary of signature names (keys) and constituent genes (values), with gene
        names matching `adata.var_names`
    sig_subset : list, Optional (default=`None`)
        Subset of keys from `signatures_dict` to attempt scoring. If `None`, score all
        signatures from `signatures_dict`
    layer : str, Optional (default=`None`)
        Layer key from `adata.layers` to use for scoring. If `None`, use `adata.X`.

    Returns
    -------
    `adata` is edited with signature scores added to `.obs`

    failed_sigs : list
        List of signatures not scored properly
    scored_sigs : list
        List of signatures scored successfully
    """
    if layer is not None:
        # put desired layer in .X for scoring
        adata.X = adata.layers[layer].copy()
    if sig_subset is None:
        sig_subset = list(signatures_dict.keys())
    else:
        assert np.all(
            [x in signatures_dict.keys() for x in sig_subset]
        ), "All keys in sig_subset must be present in signatures_dict!"
    print("Scoring {} signatures from dictionary:\n".format(len(sig_subset)))
    failed_sigs = []  # keep track of signatures that don't score properly
    scored_sigs = []  # keep track of signatures successfully scored
    for sig in sig_subset:
        # if there's an "Up" and "Down" portion to a signature, score accordingly
        if sig.lower().endswith("_up"):
            try:
                print(sig.replace("_Up", "").replace("_up", ""))
                sc.tl.score_genes(
                    adata,
                    gene_list=signatures_dict[sig],
                    gene_pool=signatures_dict[sig]
                    + signatures_dict[
                        sig.replace("_Up", "_Down").replace("_up", "_down")
                    ],
                    ctrl_size=len(
                        signatures_dict[
                            sig.replace("_Up", "_Down").replace("_up", "_down")
                        ]
                    ),
                    score_name=sig.replace("_Up", "").replace("_up", ""),
                )
                scored_sigs.append(sig.replace("_Up", "").replace("_up", ""))
            except ValueError as e:
                print(
                    "{} failed!\n\t{}".format(
                        sig.replace("_Up", "").replace("_up", ""), e
                    )
                )
                failed_sigs.append(sig.replace("_Up", "").replace("_up", ""))
        # should be able to skip the "Down" signatures as we look for "Up" lists above
        elif sig.lower().endswith("_down"):
            pass
        # otherwise, score signature normally
        else:
            try:
                print(sig)
                sc.tl.score_genes(
                    adata,
                    gene_list=signatures_dict[sig],
                    score_name=sig,
                )
                scored_sigs.append(sig)
            except ValueError as e:
                print("{} failed!\n\t{}".format(sig, e))
                failed_sigs.append(sig)
    return failed_sigs, scored_sigs


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
    key_added="rank_genes_groups",
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
    key_added : str, optional (default="rank_genes_groups")
        `adata.uns` key to add DEG results to
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
    sc.tl.rank_genes_groups(adata, groupby=groupby, layer=layer, use_raw=False, method=de_method, key_added=key_added)

    # unique groups in DEG analysis
    groups = adata.obs[groupby].unique().tolist()

    # DEGs as dictionary
    markers = signature_dict_from_rank_genes_groups(
        adata,
        uns_key=key_added,
        groups=groups,
        n_genes=n_genes,
        ambient=ambient,
    )

    # total and unique features on plot
    features = signature_dict_values(signatures_dict=markers, unique=False)

    # plotting workflow if desired
    if plot_type is not None:
        # plot dimensions (long vertical)
        if plot_type in ["dotplot", "matrixplot", "dotmatrix", "stacked_violin"]:
            figsize = (
                (len(groups) / 3) * figsize_scale,
                (len(features) / 5) * figsize_scale,
            )
        # plot dimensions (long horizontal)
        elif plot_type == "heatmap":
            figsize = (
                (len(features) / 5) * figsize_scale,
                (len(groups) / 3) * figsize_scale,
            )

        # build plot
        my_plot = custom_heatmap(
            adata,
            groupby=groupby,
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
    features = signature_dict_values(signatures_dict=markers, unique=False)
    unique_features = signature_dict_values(signatures_dict=markers, unique=True)

    print(
        "Plotting {} total features and {} unique features across {} factors".format(
            len(features), len(unique_features), len(groups)
        )
    )

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
        vars_dict=markers,
        cluster_obs=dendrogram,
        plot_type=plot_type,
        cmap=cmap,
        figsize=figsize,
        save=save_to,
        **kwargs,
    )
    return (markers, my_plot)


def scDEG_decoupler_ORA(
    adata,
    uns_key,
    net,
    max_FDRpval=0.05,
    min_logFC=np.log2(1.5),
    out_dir="./",
    save_prefix="",
    save_output=False,
    get_ora_df_kwargs={},
    decoupler_dotplot_facet_kwargs={},
):
    """
    Quickly process DEGs from scRNA dataset through `decoupler` ORA and create plot for
    initial look at pathways

    `decoupler` tools used:
        * "ORA" for biological pathways

    Parameters
    ----------
    adata : anndata.AnnData
        Object containing rank_genes_groups results in `adata.uns`
    uns_key : str
        Key to `adata.uns` containing rank_genes_groups results
    net : pd.DataFrame
        Network dataframe required to run ORA. e.g. `msigdb` where `msigdb` is a
        `pd.DataFrame` from `dc.get_resource`.
    max_FDRpval : float, optional (default=0.05)
        FDR p-value cutoff for using genes and plotting significant ORA pathways.
    min_logFC : float, optional (default=1.5)
        logFC cutoff for using genes and plotting significant ORA pathways.
    out_dir : str, optional (default="./")
        Path to directory to save plots to
    save_prefix : str, optional (default="")
        String to prepend to output plots to make names unique
    save_output : bool, optional (default=`False`)
        If `True`, save output dataframe to `out_dir/save_prefix_NMF_ORA.csv`
    get_ora_df_kwargs : dict, optional (default={})
        Keyword arguments to pass to `dc.get_ora_df`. e.g. `source`, `target`,
        `n_background`.
    decoupler_dotplot_facet_kwargs : dict, optional (default={})
        Keyword arguments to pass to `decoupler_dotplot_facet`. e.g. `top_n`, `cmap`,
        `dpi`

    Returns
    -------
    enr_pvals : pd.DataFrame
        ORA output as dataframe

    Saves `decoupler` ORA dotplots to `out_dir/`.
    """
    # get DEGs from all comparisons
    degs = sc.get.rank_genes_groups_df(adata, group=None, key=uns_key).set_index(
        "names"
    )
    if "group" not in degs.columns:
        print(f"No 'group' column found. Adding as {uns_key}.")
        degs["group"] = uns_key
    # subset to significant DEGs
    degs = degs.loc[
        (degs.pvals_adj <= max_FDRpval) & (abs(degs.logfoldchanges) >= min_logFC), :
    ]

    # perform over representation analysis
    # iterate through comparisons and perform ORA
    enr_pvals = pd.DataFrame()  # initialize df for outputs
    for group in degs.group.unique():
        try:
            print(group)
            # get one group at a time
            tmp_ora = degs.loc[degs.group == group].copy()
            # perform ORA
            ora = dc.get_ora_df(
                df=tmp_ora,
                net=net,
                verbose=True,
                **get_ora_df_kwargs,
            )
            # subset to significant terms
            ora = ora.loc[ora["FDR p-value"] <= max_FDRpval]
            ora["group"] = group
            ora["reference"] = "rest"
            enr_pvals = pd.concat([enr_pvals, ora])
        except:
            print(f"Error in {group}!")

    enr_pvals = enr_pvals.reset_index(drop=True)
    if len(enr_pvals) > 0:
        # re-format enr_pvals for outputs
        enr_pvals["LE_size"] = enr_pvals.Features.str.split(";").apply(len)
        enr_pvals["-log10(FDR pval)"] = -np.log10(enr_pvals["FDR p-value"])
        enr_pvals.rename(columns={"Overlap ratio": "LE Proportion"}, inplace=True)

        # plot ORA results
        decoupler_dotplot_facet(
            df=enr_pvals,
            group_col="group",
            x="Combined score",
            y="Term",
            c="-log10(FDR pval)",
            s="LE Proportion",
            save=f"{out_dir}/{save_prefix}DEG_ORA.png",
            **decoupler_dotplot_facet_kwargs,
        )

        if save_output:
            enr_pvals.to_csv(f"{out_dir}/{save_prefix}_DEG_ORA.csv")
            print(f"Saved output dataframe to {out_dir}/{save_prefix}_DEG_ORA.csv")

    print("Done!")
    return enr_pvals


def NMF_decoupler_ORA(
    adata,
    net,
    top_n_genes=None,
    max_FDRpval=0.05,
    out_dir="./",
    save_prefix="",
    save_output=False,
    get_ora_df_kwargs={},
    decoupler_dotplot_facet_kwargs={},
):
    """
    Quickly process cNMF loadings through `decoupler` ORA and create plot for
    initial look at pathways

    `decoupler` tools used:
        * "ORA" for biological pathways

    Parameters
    ----------
    adata : anndata.AnnData
        Object containing cNMF results in `adata.uns["cnmf_markers"]`
    net : pd.DataFrame
        Network dataframe required to run ORA. e.g. `msigdb` where `msigdb` is a
        `pd.DataFrame` from `dc.get_resource`.
    top_n_genes : int or `None`, optional (default=`None`)
        If `None`, use entire `adata.uns["cnmf_markers"]` dataframe. If an integer,
        select first `top_n_genes` rows of `adata.uns["cnmf_markers"]`.
    max_FDRpval : float or `None`, optional (default=0.05)
        FDR p-value cutoff for plotting significant ORA pathways. If `None`, don't
        filter and show top 20 terms by FDR p-value.
    out_dir : str, optional (default="./")
        Path to directory to save plots to
    save_prefix : str, optional (default="")
        String to prepend to output plots to make names unique
    save_output : bool, optional (default=`False`)
        If `True`, save output dataframe to `out_dir/save_prefix_NMF_ORA.csv`
    get_ora_df_kwargs : dict, optional (default={})
        Keyword arguments to pass to `dc.get_ora_df`. e.g. `source`, `target`,
        `n_background`.
    decoupler_dotplot_facet_kwargs : dict, optional (default={})
        Keyword arguments to pass to `decoupler_dotplot_facet`. e.g. `top_n`, `cmap`,
        `dpi`

    Returns
    -------
    enr_pvals : pd.DataFrame
        ORA output as dataframe

    Saves `decoupler` ORA dotplots to `out_dir/`.
    """
    # get loadings in long form (top 30 markers)
    loadings = pd.melt(
        adata.uns["cnmf_markers"], var_name="usage", value_name="genesymbol"
    ).set_index("genesymbol")
    if top_n_genes is not None:
        if top_n_genes < len(loadings):
            loadings = loadings.iloc[: top_n_genes + 1, :].copy()

    # perform over representation analysis
    # iterate through comparisons and perform ORA
    enr_pvals = pd.DataFrame()  # initialize df for outputs
    for group in loadings.usage.unique():
        print(group)
        # get one group at a time
        tmp_ora = loadings.loc[loadings.usage == group].copy()
        # perform ORA
        ora = dc.get_ora_df(
            df=tmp_ora,
            net=net,
            verbose=True,
            **get_ora_df_kwargs,
        )
        # subset to significant terms
        ora = ora.loc[ora["FDR p-value"] <= max_FDRpval]
        ora["group"] = group
        ora["reference"] = "rest"
        enr_pvals = pd.concat([enr_pvals, ora])

    enr_pvals = enr_pvals.reset_index(drop=True)
    if len(enr_pvals) > 0:
        # re-format enr_pvals for outputs
        enr_pvals["LE_size"] = enr_pvals.Features.str.split(";").apply(len)
        enr_pvals["-log10(FDR pval)"] = -np.log10(enr_pvals["FDR p-value"])
        enr_pvals.rename(columns={"Overlap ratio": "LE Proportion"}, inplace=True)

        # plot ORA results
        decoupler_dotplot_facet(
            df=enr_pvals,
            group_col="group",
            x="Combined score",
            y="Term",
            c="-log10(FDR pval)",
            s="LE Proportion",
            save=f"{out_dir}/{save_prefix}NMF_ORA.png",
            **decoupler_dotplot_facet_kwargs,
        )

        if save_output:
            enr_pvals.to_csv(f"{out_dir}/{save_prefix}_NMF_ORA.csv")
            print(f"Saved output dataframe to {out_dir}/{save_prefix}_NMF_ORA.csv")

    print("Done!")
    return enr_pvals
