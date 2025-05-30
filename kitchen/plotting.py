# -*- coding: utf-8 -*-
"""
Custom plotting functions for data in `pd.DataFrame`, `anndata.AnnData`, and
`decoupler` formats
"""
import os
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

from math import ceil
from anndata import AnnData
from scanpy.get import obs_df
from matplotlib import patheffects as pe
from matplotlib.axes import Axes
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.markers import TICKDOWN, TICKLEFT
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from mycolorpy import colorlist as mcp
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list

from .ingredients import signature_dict_values

sc.set_figure_params(frameon=False, dpi=100, dpi_save=200, format="png")


def myround(x, n_precision=3):
    """Custom rounding function"""
    # round to nearest whole number first
    xx = np.round(x, 0)
    # if any x round to zero at nearest whole number,
    # round those to n_precision decimal places
    if sum(xx == 0) > 0:
        xx[xx == 0] = np.round(np.array(x)[xx == 0], n_precision)
    return xx


def mylog1p(x, base=2, pseudocount=0.1):
    """Custom log1p function"""
    return np.emath.logn(base, np.array(x) + pseudocount)


def myexpm1(x, base=2, pseudocount=0.1, myround=False):
    """Custom expm1 function"""
    if myround:
        return myround(np.power(base, x) - pseudocount)
    else:
        return np.power(base, x) - pseudocount


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


def save_plot(fig, ax, save):
    if save is not None:
        if ax is not None:
            if fig is not None:
                fig.savefig(save, bbox_inches="tight")
            else:
                raise ValueError("fig is `None`, cannot save figure.")
        else:
            raise ValueError("ax is `None`, cannot save figure.")


def pie_from_col(df, col, title=None, figsize=(8, 8), save_to=None):
    """
    Create a pie chart from the values in a pd.DataFrame column

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe from which to plot
    col : str
        Column in `df` to create pie chart for
    title : str
        Title of plot
    figsize : tuple of float, optional (default=(8,8))
        Size of figure
    save_to : str, optional (default=`None`)
        Path to image file to save plot to
    """

    def label_function(val):
        return f"{val / 100 * len(df):.0f} ({val:.0f}%)"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    df.groupby(col).size().plot(
        kind="pie",
        autopct=label_function,
        title=title,
        textprops={"fontsize": 14},
        ax=ax,
    )
    ax.set_title(ax.get_title(), fontsize=14)
    if save_to is None:
        return fig
    else:
        fig.tight_layout()
        fig.savefig(save_to)


def significance_bar(
    start,
    end,
    height,
    displaystring,
    linewidth=1.2,
    markersize=8,
    boxpad=0.3,
    fontsize=15,
    color="k",
    ax=None,
    horizontal=False,
):
    """
    Draw significance bracket on matplotlib figure
    """
    # for horizontal boxplots
    if horizontal:
        # draw a line with downticks at the ends
        if ax is None:
            plt.plot(
                [height] * 2,
                [start, end],
                "-",
                color=color,
                lw=linewidth,
                marker=TICKLEFT,
                markeredgewidth=linewidth,
                markersize=markersize,
            )
            # draw the text with a bounding box covering up the line
            plt.text(
                height,
                0.5 * (start + end),
                displaystring,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="none",
                    edgecolor="none",
                    boxstyle="Circle,pad=" + str(boxpad),
                ),
                size=fontsize,
            )
        else:
            ax.plot(
                [height] * 2,
                [start, end],
                "-",
                color=color,
                lw=linewidth,
                marker=TICKLEFT,
                markeredgewidth=linewidth,
                markersize=markersize,
            )
            # draw the text with a bounding box covering up the line
            ax.text(
                height,
                0.5 * (start + end),
                displaystring,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="none",
                    edgecolor="none",
                    boxstyle="Circle,pad=" + str(boxpad),
                ),
                size=fontsize,
            )

    # for vertical boxplots
    else:
        # draw a line with downticks at the ends
        if ax is None:
            plt.plot(
                [start, end],
                [height] * 2,
                "-",
                color=color,
                lw=linewidth,
                marker=TICKDOWN,
                markeredgewidth=linewidth,
                markersize=markersize,
            )
            # draw the text with a bounding box covering up the line
            plt.text(
                0.5 * (start + end),
                height,
                displaystring,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="none",
                    edgecolor="none",
                    boxstyle="Circle,pad=" + str(boxpad),
                ),
                size=fontsize,
            )
        else:
            ax.plot(
                [start, end],
                [height] * 2,
                "-",
                color=color,
                lw=linewidth,
                marker=TICKDOWN,
                markeredgewidth=linewidth,
                markersize=markersize,
            )
            # draw the text with a bounding box covering up the line
            ax.text(
                0.5 * (start + end),
                height,
                displaystring,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="none",
                    edgecolor="none",
                    boxstyle="Circle,pad=" + str(boxpad),
                ),
                size=fontsize,
            )


def calc_significance(
    df,
    ax,
    feature,
    groupby,
    groupby_order=None,
    test_pairs=None,
    method="t-test",
    transform=None,
    bonferroni=False,
    log_scale=None,
    **kwargs,
):
    """
    Calculate significance with t-test between two groups

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing data to plot
    ax : matplotlib.axes
        Axes object to plot significance bars on
    feature : str
        DataFrame column to plot (y variable)
    groupby : list of str
        Columns from `a` or `a.obs` to group by (x variable)
    groupby_order : list of str, optional (default=`None`)
        List of values in `df[groupby]` specifying the order of groups on x-axis.
    test_pairs : list of tuple, optional (default=`None`)
        List of pairs of `df[groupby]` values to include in testing. If `None`, test all
        levels in `df[groupby]` pairwise.
    method : literal ["t-test","wilcoxon"], optional (default="t-test")
        Method for performing significance testing across groups
    transform : literal ["log1p","log10p","log2p","arcsinh",None], optional (default=`None`)
        Transformation to apply to values before testing. "log1p"=np.log(val+1),
        "log10p"=np.log10(val+1), "log2p"=np.log2(val+1), "arcsinh"=np.arcsinh(val).
    bonferroni : bool, optional (default=False)
        Adjust significance p-values with simple Bonferroni correction
    log_scale : int, optional (default=`None`)
        Set axis scale(s) to log. Numeric values are interpreted as the desired base
        (e.g. 10). When `None`, plot defers to the existing Axes scale.
    **kwargs : optional
        Keyword args to pass to `significance_bar`

    Returns
    -------
    sig_out : dict
        Dictionary of t-test statistics if `sig==True`. Otherwise, write to `.csv` in
        `outdir/`.
    """
    assert method in [
        "t-test",
        "wilcoxon",
    ], "method must be one of ['t-test','wilcoxon']"
    if transform is not None:
        assert transform in [
            "log1p",
            "log10p",
            "log2p",
            "arcsinh",
        ], "transform must be one of ['log1p','log10p','log2p','arcsinh',None]"
    # initialize dictionary of p values
    sig_out = {
        "variable": [],
        "group1": [],
        "group2": [],
        "pvalue": [],
        "pvalue_adj": [],
    }
    # initialize significant count for bar height
    sig_count = 0

    # determine how to loop through 'groupby'
    if groupby_order is not None:
        indexer = groupby_order
    elif pd.api.types.is_categorical_dtype(df[groupby]):
        indexer = df[groupby].cat.categories
    else:
        indexer = df[groupby].unique()

    for i_sig in range(len(indexer)):
        for i_sig_2 in [x for x in range(i_sig, len(indexer)) if x != i_sig]:
            # check if (indexer[i_sig], indexer[i_sig_2]) in test_pairs
            if test_pairs is not None:
                if ((indexer[i_sig], indexer[i_sig_2]) in test_pairs) | (
                    (indexer[i_sig_2], indexer[i_sig]) in test_pairs
                ):
                    skip = False
                else:
                    skip = True
            else:
                skip = False

            if not skip:
                # if label has more than two classes, automatically perform
                # t-tests and add significance bars to plots
                if method == "t-test":
                    if transform is None:
                        _, p_value = stats.ttest_ind(
                            df.loc[df[groupby] == indexer[i_sig], feature].dropna(),
                            df.loc[df[groupby] == indexer[i_sig_2], feature].dropna(),
                        )
                    elif transform == "log1p":
                        _, p_value = stats.ttest_ind(
                            np.log1p(
                                df.loc[df[groupby] == indexer[i_sig], feature].dropna()
                            ),
                            np.log1p(
                                df.loc[
                                    df[groupby] == indexer[i_sig_2], feature
                                ].dropna()
                            ),
                        )
                    elif transform == "log10p":
                        _, p_value = stats.ttest_ind(
                            np.log10(
                                df.loc[df[groupby] == indexer[i_sig], feature].dropna()
                                + 1
                            ),
                            np.log10(
                                df.loc[
                                    df[groupby] == indexer[i_sig_2], feature
                                ].dropna()
                                + 1
                            ),
                        )
                    elif transform == "log2p":
                        _, p_value = stats.ttest_ind(
                            np.log2(
                                df.loc[df[groupby] == indexer[i_sig], feature].dropna()
                                + 1
                            ),
                            np.log2(
                                df.loc[
                                    df[groupby] == indexer[i_sig_2], feature
                                ].dropna()
                                + 1
                            ),
                        )
                    elif transform == "arcsinh":
                        _, p_value = stats.ttest_ind(
                            np.arcsinh(
                                df.loc[df[groupby] == indexer[i_sig], feature].dropna()
                            ),
                            np.arcsinh(
                                df.loc[
                                    df[groupby] == indexer[i_sig_2], feature
                                ].dropna()
                            ),
                        )
                elif method == "wilcoxon":
                    if transform is None:
                        _, p_value = stats.mannwhitneyu(
                            df.loc[df[groupby] == indexer[i_sig], feature].dropna(),
                            df.loc[df[groupby] == indexer[i_sig_2], feature].dropna(),
                        )
                    elif transform == "log1p":
                        _, p_value = stats.mannwhitneyu(
                            np.log1p(
                                df.loc[df[groupby] == indexer[i_sig], feature].dropna()
                            ),
                            np.log1p(
                                df.loc[
                                    df[groupby] == indexer[i_sig_2], feature
                                ].dropna()
                            ),
                        )
                    elif transform == "log10p":
                        _, p_value = stats.mannwhitneyu(
                            np.log10(
                                df.loc[df[groupby] == indexer[i_sig], feature].dropna()
                                + 1
                            ),
                            np.log10(
                                df.loc[
                                    df[groupby] == indexer[i_sig_2], feature
                                ].dropna()
                                + 1
                            ),
                        )
                    elif transform == "log2p":
                        _, p_value = stats.mannwhitneyu(
                            np.log2(
                                df.loc[df[groupby] == indexer[i_sig], feature].dropna()
                                + 1
                            ),
                            np.log2(
                                df.loc[
                                    df[groupby] == indexer[i_sig_2], feature
                                ].dropna()
                                + 1
                            ),
                        )
                    elif transform == "arcsinh":
                        _, p_value = stats.mannwhitneyu(
                            np.arcsinh(
                                df.loc[df[groupby] == indexer[i_sig], feature].dropna()
                            ),
                            np.arcsinh(
                                df.loc[
                                    df[groupby] == indexer[i_sig_2], feature
                                ].dropna()
                            ),
                        )
                # Bonferroni correction
                pvalue_adj = p_value * len(indexer)
                # dump results into dictionary
                sig_out["variable"].append(feature)
                sig_out["group1"].append(indexer[i_sig])
                sig_out["group2"].append(indexer[i_sig_2])
                sig_out["pvalue"].append(p_value)
                sig_out["pvalue_adj"].append(pvalue_adj)
                # use adj p-value if 'bonferroni' is True
                tester = pvalue_adj if bonferroni else p_value
                if tester <= 0.05:
                    sig_count += 1  # increment significant count
                    if tester < 0.0001:
                        displaystring = r"***"
                    elif tester < 0.001:
                        displaystring = r"**"
                    else:
                        displaystring = r"*"
                    # offset by 15 percent for each significant pair
                    # or 25 percent if log scale
                    if log_scale is None:
                        height = (
                            df[feature].max() + 0.15 * df[feature].max() * sig_count
                        )
                    else:
                        height = (
                            df[feature].max()
                            + 0.1
                            * df[feature].max()
                            * sig_count
                            * sig_count
                            * log_scale
                        )
                    # set up significance bar
                    bar_centers = np.array([i_sig, i_sig_2])
                    significance_bar(
                        bar_centers[0],
                        bar_centers[1],
                        height,
                        displaystring,
                        ax=ax,
                        **kwargs,
                    )
    return sig_out


def calc_significance_split(
    df,
    ax,
    feature,
    groupby,
    splitby,
    method="t-test",
    transform=None,
    groupby_order=None,
    splitby_order=None,
    log_scale=None,
    **kwargs,
):
    """
    Calculate significance with t-test between two groups

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing data to plot
    ax : matplotlib.axes
        Axes object to plot significance bars on
    feature : str
        DataFrame column to plot (y variable)
    groupby : str
        Column from `dfa` to group by (x variable)
    splitby : str
        Categorical column from `df` to split boxes/violins by
    method : literal ["t-test","wilcoxon"], optional (default="t-test")
        Method for performing significance testing across groups
    transform : literal ["log1p","log10p","log2p","arcsinh",None], optional (default=`None`)
        Transformation to apply to values before testing. "log1p"=np.log(val+1),
        "log10p"=np.log10(val+1), "log2p"=np.log2(val+1), "arcsinh"=np.arcsinh(val).
    groupby_order : list of str, optional (default=`None`)
        List of values in `df[groupby]` specifying the order of groups on x-axis.
    splitby_order : list of str, optional (default=`None`)
        Order of categories in `df[splitby]`
    log_scale : int, optional (default=`None`)
        Set axis scale(s) to log. Numeric values are interpreted as the desired base
        (e.g. 10). When `None`, plot defers to the existing Axes scale.
    **kwargs : optional
        Keyword args to pass to `significance_bar`

    Returns
    -------
    sig_out : dict
        Dictionary of t-test statistics if `sig==True`. Otherwise, write to `.csv` in
        `outdir/`.
    """
    # initialize dictionary of p values
    sig_out = {
        "variable": [],
        "group1": [],
        "group2": [],
        "pvalue": [],
        "pvalue_adj": [],
    }

    # determine how to loop through 'groupby'
    if groupby_order is not None:
        indexer = groupby_order
    elif pd.api.types.is_categorical_dtype(df[groupby]):
        indexer = df[groupby].cat.categories
    else:
        indexer = df[groupby].unique()
    # determine how to loop through 'splitby'
    if splitby_order is not None:
        jndexer = splitby_order
    elif pd.api.types.is_categorical_dtype(df[splitby]):
        jndexer = df[splitby].cat.categories
    else:
        jndexer = df[splitby].unique()

    assert len(jndexer) == 2, "Can only calculate significance across a binary splitby"

    # loop through `groupby`
    for i_sig in range(len(indexer)):
        # initialize significant count for bar height
        sig_count = 0
        # perform t-tests and add significance bars to plots
        if method == "t-test":
            if transform is None:
                _, p_value = stats.ttest_ind(
                    df.loc[
                        (df[groupby] == indexer[i_sig]) & (df[splitby] == jndexer[0]),
                        feature,
                    ].dropna(),
                    df.loc[
                        (df[groupby] == indexer[i_sig]) & (df[splitby] == jndexer[1]),
                        feature,
                    ].dropna(),
                )
            elif transform == "log1p":
                _, p_value = stats.ttest_ind(
                    np.log1p(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[0]),
                            feature,
                        ].dropna()
                    ),
                    np.log1p(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[1]),
                            feature,
                        ].dropna()
                    ),
                )
            elif transform == "log10p":
                _, p_value = stats.ttest_ind(
                    np.log10(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[0]),
                            feature,
                        ].dropna()
                        + 1
                    ),
                    np.log10(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[1]),
                            feature,
                        ].dropna()
                        + 1
                    ),
                )
            elif transform == "log2p":
                _, p_value = stats.ttest_ind(
                    np.log2(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[0]),
                            feature,
                        ].dropna()
                        + 1
                    ),
                    np.log2(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[1]),
                            feature,
                        ].dropna()
                        + 1
                    ),
                )
            elif transform == "arcsinh":
                _, p_value = stats.ttest_ind(
                    np.arcsinh(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[0]),
                            feature,
                        ].dropna()
                    ),
                    np.arcsinh(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[1]),
                            feature,
                        ].dropna()
                    ),
                )
        elif method == "wilcoxon":
            if transform is None:
                _, p_value = stats.mannwhitneyu(
                    df.loc[
                        (df[groupby] == indexer[i_sig]) & (df[splitby] == jndexer[0]),
                        feature,
                    ].dropna(),
                    df.loc[
                        (df[groupby] == indexer[i_sig]) & (df[splitby] == jndexer[1]),
                        feature,
                    ].dropna(),
                )
            elif transform == "log1p":
                _, p_value = stats.mannwhitneyu(
                    np.log1p(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[0]),
                            feature,
                        ].dropna()
                    ),
                    np.log1p(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[1]),
                            feature,
                        ].dropna()
                    ),
                )
            elif transform == "log10p":
                _, p_value = stats.mannwhitneyu(
                    np.log10(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[0]),
                            feature,
                        ].dropna()
                        + 1
                    ),
                    np.log10(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[1]),
                            feature,
                        ].dropna()
                        + 1
                    ),
                )
            elif transform == "log2p":
                _, p_value = stats.mannwhitneyu(
                    np.log2(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[0]),
                            feature,
                        ].dropna()
                        + 1
                    ),
                    np.log2(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[1]),
                            feature,
                        ].dropna()
                        + 1
                    ),
                )
            elif transform == "arcsinh":
                _, p_value = stats.mannwhitneyu(
                    np.arcsinh(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[0]),
                            feature,
                        ].dropna()
                    ),
                    np.arcsinh(
                        df.loc[
                            (df[groupby] == indexer[i_sig])
                            & (df[splitby] == jndexer[1]),
                            feature,
                        ].dropna()
                    ),
                )

        # dump results into dictionary
        sig_out["variable"].append(feature)
        sig_out["group1"].append(str(indexer[i_sig]) + " " + str(jndexer[0]))
        sig_out["group2"].append(str(indexer[i_sig]) + " " + str(jndexer[1]))
        sig_out["pvalue"].append(p_value)
        sig_out["pvalue_adj"].append(p_value)
        if p_value <= 0.05:
            sig_count += 1  # increment significant count
            if p_value < 0.0001:
                displaystring = r"***"
            elif p_value < 0.001:
                displaystring = r"**"
            else:
                displaystring = r"*"
            # offset by 15 percent for each significant pair
            # or 25 percent if log scale
            if log_scale is None:
                height = (
                    df.loc[(df[groupby] == indexer[i_sig]), feature].max()
                    + 0.15
                    * df.loc[(df[groupby] == indexer[i_sig]), feature].max()
                    * sig_count
                )
            else:
                height = (
                    df.loc[(df[groupby] == indexer[i_sig]), feature].max()
                    + 0.1
                    * df.loc[(df[groupby] == indexer[i_sig]), feature].max()
                    * sig_count
                    * sig_count
                    * log_scale
                )
            # set up significance bar
            bar_centers = np.array([i_sig - 0.25, i_sig + 0.25])
            significance_bar(
                bar_centers[0], bar_centers[1], height, displaystring, ax=ax, **kwargs
            )
    return sig_out


def build_gridspec(panels, ncols, panelsize=(3, 3)):
    """
    Create `gridspec.GridSpec` object from a list of panels

    Parameters
    ----------
    panels : list or str
        List of panels in plot grid. If string, only one panel is made.
    ncols : int
        Number of columns in grid. Number of rows is calculated from `len(panels)` and
        `ncols`.
    panelsize : tuple of float, optional (default=(3,3))
        Size in inches of each panel within the plot grid.

    Returns
    -------
    gs : gridspec.GridSpec
        GridSpec object
    fig : matplotlib.Figure
        Figure object
    """
    # coerce single string to list for looping
    if isinstance(panels, str):
        panels = [panels]
    # calculate number of panels for faceting
    n_panels = len(panels) if isinstance(panels, list) else 1
    # determine number of rows and columns
    if n_panels <= ncols:
        n_rows, n_cols = 1, n_panels
    else:
        n_rows, n_cols = ceil(n_panels / ncols), ncols
    # determine size of figure
    fig = plt.figure(figsize=(n_cols * panelsize[0], n_rows * panelsize[1]))
    left, bottom = 0.1 / n_cols, 0.1 / n_rows
    # build gs object
    gs = gridspec.GridSpec(
        nrows=n_rows,
        ncols=n_cols,
        wspace=0.1,
        left=left,
        bottom=bottom,
        right=1 - (n_cols - 1) * left - 0.01 / n_cols,
        top=1 - (n_rows - 1) * bottom - 0.1 / n_rows,
    )
    return gs, fig


def split_violin(
    a,
    features,
    groupby=None,
    groupby_order=None,
    splitby=None,
    splitby_order=None,
    points_colorby=None,
    points_colordict=None,
    layer=None,
    log_scale=None,
    log_scale_method="functionlog",
    pseudocount=1.0,
    plot_type="violin",
    scale="width",
    sig=False,
    test_method="t-test",
    test_transform=None,
    split=True,
    strip=True,
    jitter=True,
    size=3,
    hlines=None,
    panelsize=(3, 3),
    ncols=1,
    sample_n=False,
    ylabel=None,
    xlab_rot=None,
    titles=None,
    legend=True,
    save=None,
    dpi=300,
):
    """
    Plot genes grouped by one variable and split by another

    Parameters
    ----------
    a : Union[anndata.Anndata, pd.DataFrame]
        The annotated data matrix of shape `n_obs` by `n_vars`. Rows correspond to
        samples and columns to genes. Can also be `pd.DataFrame`.
    features : list of str
        List of genes, `.obs` columns, or DataFrame columns to plot (if `a` is
        `pd.DataFrame`).
    groupby : str, optional (default=`None`)
        Column from `a` or `a.obs` to group by (x variable)
    groupby_order : list of str, optional (default=`None`)
        List of values in `a[groupby]` or `a.obs[groupby]` specifying the order of
        groups on x-axis. If `groupby` is a list, `groupby_order` should also be a list
        with corresponding orders in each element.
    splitby : str, optional (default=`None`)
        Categorical `.obs` column to split violins by.
    splitby_order : list of str, optional (default=`None`)
        Order of categories in `adata.obs[splitby]`.
    points_colorby : str, optional (default=`None`)
        Categorical `.obs` column to color stripplot points by.
    points_colordict : dict, optional (default=`None`)
        Dictionary of group, color pairs from `points_colorby` to color points by
    layer : str, optional (default=`None`)
        Key from `layers` attribute of `adata` if present
    log_scale : int, optional (default=`None`)
        Set axis scale(s) to log. Numeric values are interpreted as the desired base
        (e.g. 10). When `None`, plot defers to the existing Axes scale.
    log_scale_method : literal ['functionlog','symlog'], optional (default='functionlog')
        Method for adjusting y-axis scale if `log_scale` is not `None`
    pseudocount : float, optional (default=1.0)
        Pseudocount to add to values before log-transforming with base=`log_scale`
    plot_type : str, optional (default="violin")
        "violin" for violinplot, "box" for boxplot
    scale : str, optional (default="width")
        See :func:`~seaborn.violinplot`.
    sig : bool, optional (default=`False`)
        Perform significance testing (2-way t-test) between all groups across split_by
        and add significance bars to plot(s)
    test_method : literal ["t-test","wilcoxon"], optional (default="t-test")
        Method for performing significance testing across groups
    test_transform : literal ["log1p","log10p","log2p","arcsinh"], optional (default=`None`)
        Transformation to apply to values before testing. "log1p"=np.log(val+1),
        "log10p"=np.log10(val+1), "log2p"=np.log2(val+1), "arcsinh"=np.arcsinh(val).
    split : bool, optional (default=`True`)
        Split box/violinplots and dodge stripplots if `splitby` is not `None`
    strip : bool, optional (default=`True`)
        Show a strip plot on top of the violin plot.
    jitter : Union[int, float, bool], optional (default=`True`)
        If set to 0, no points are drawn. See :func:`~seaborn.stripplot`.
    size : int, optional (default=3)
        Size of the jitter points
    hlines : dict, optional (default=`None`)
        Dictionary containing values to draw horizontal lines at for each feature.
        e.g. `{"feature1":10,"feature2":12,"feature3":8}` where "feature1", etc. are
        found in `features`.
    panelsize : tuple of int, optional (default=(3, 3))
        Size of each panel in output figure in inches
    ncols : int, optional (default=1)
        Number of columns in gridspec. If `None` use `len(features)`.
    sample_n : bool, optional (default=False)
        Add sample number to x-axis tick labels for each group
    ylabel : str, optional (default=`None`)
        Label for y axes. If `None` use `colors`.
    xlab_rot : int, optional (default=`None`)
        Rotation (in degrees) to force on all x ticklabels. If `None`, auto-rotate x
        ticklabels 90 degrees if they're longer than 10 characters.
    titles : list of str, optional (default=`None`)
        Titles for each set of axes. If `None` use `features`.
    legend : bool, optional (default=`True`)
        Add legend to plot
    save : str, optional (default=`None`)
        Path to file to save image to. If `None`, return axes objects
    dpi : float, optional (default=300)
        Resolution in dots per inch for saving figure. Ignored if `save` is `None`.

    Returns
    -------
    fig : matplotlib.Figure
        Return figure object if `save==None`. Otherwise, write to `save`.
    sig_out : dict
        Dictionary of t-test statistics if `sig==True`. Otherwise, write to `.csv` in
        `outdir/`.
    """
    # coerce single string to list for looping
    if isinstance(features, str):
        features = [features]
    # prep df for plotting
    if isinstance(a, AnnData):
        # get genes of interest
        extra_cols = []
        if splitby is not None:
            extra_cols.append(splitby)
        if points_colorby is not None:
            extra_cols.append(points_colorby)
        if groupby is not None:
            extra_cols.append(groupby)
        # remove repetitive cols
        extra_cols = [x for x in list(set(extra_cols)) if x not in features]
        df = obs_df(
            a,
            features + extra_cols if len(extra_cols) > 0 else features,
            layer=layer,
        )
        new_gene_names = [x for x in features if x not in list(set(extra_cols))]
    elif isinstance(a, pd.DataFrame):
        df = a.copy()
        new_gene_names = features
    else:
        raise ValueError("'a' must be AnnData object or pandas DataFrame")

    # add hue to plotting df
    if splitby is None:
        df["hue"] = 0
    else:
        df["hue"] = df[splitby].astype(str).values
        df["hue"] = df["hue"].astype("category")
        splitby = "hue"  # set to 'hue' for plotting
        # splitby_order
        if splitby_order is not None:
            # check if given splitby_order contains all categories
            if all([j in df["hue"].cat.categories for j in splitby_order]):
                # reorder categories with manual splitby_order
                df["hue"] = df["hue"].cat.set_categories(splitby_order, ordered=True)
            else:
                # ignore splitby_order due to mismatch
                print(
                    "\tElements in `splitby_order` do not match `splitby` categories... ignoring manual ordering"
                )
                # set order using cat.categories for downstream
                splitby_order = list(df["hue"].cat.categories)
        else:
            # if no splitby_order given, set order using cat.categories for downstream
            splitby_order = list(df["hue"].cat.categories)

    # add point hue
    if points_colorby is None:
        df["points_hue"] = 0
    else:
        df["points_hue"] = df[points_colorby].astype(str).values
        df["points_hue"] = df["points_hue"].astype("category")
        points_colorby = "points_hue"  # set to 'points_hue' for plotting

    # add group
    if groupby is None:
        df["group"] = 0
        groupby_order = [0]
    else:
        df["group"] = df[groupby].astype(str).values
        df["group"] = df["group"].astype("category")
        # groupby_order
        if groupby_order is not None:
            # check if given groupby_order contains all categories
            if all([j in df["group"].cat.categories for j in groupby_order]):
                # reorder categories with manual groupby_order
                df["group"] = df["group"].cat.set_categories(
                    groupby_order, ordered=True
                )
            else:
                # ignore groupby_order due to mismatch
                print(
                    "\tElements in `groupby_order` do not match `groupby` categories... ignoring manual ordering"
                )
                # set order using cat.categories for downstream
                groupby_order = list(df["group"].cat.categories)
        else:
            # if no groupby_order given, set order using cat.categories for downstream
            groupby_order = list(df["group"].cat.categories)

    df_tidy = pd.melt(
        df, id_vars=["hue", "points_hue", "group"], value_vars=new_gene_names
    )

    # seaborn plot style
    with sns.axes_style("whitegrid"):
        # generate gs object
        gs, fig = build_gridspec(
            panels=new_gene_names, ncols=ncols, panelsize=panelsize
        )
        if sig:
            if len(df[splitby].unique()) == 2:
                # initialize df of p values
                sig_out = pd.DataFrame(
                    {
                        "variable": [],
                        "group1": [],
                        "group2": [],
                        "pvalue": [],
                        "pvalue_adj": [],
                    }
                )
            else:
                print("WARNING: Can only calculate significance for binary splitby")
                sig = False

        for iv, variable in enumerate(new_gene_names):
            tmp = df_tidy.loc[df_tidy["variable"] == variable, :].copy()

            # add subplot to gs object
            _ax = plt.subplot(gs[iv])

            if plot_type == "violin":
                sns.violinplot(
                    x="group",
                    y="value",
                    data=tmp,
                    inner=None,
                    hue=splitby if splitby is not None else "group",
                    hue_order=splitby_order if splitby is not None else groupby_order,
                    split=False if splitby is None else split,
                    scale=scale,
                    orient="vertical",
                )
            elif plot_type == "box":
                sns.boxplot(
                    x="group",
                    y="value",
                    data=tmp,
                    hue=splitby if splitby is not None else "group",
                    hue_order=splitby_order if splitby is not None else groupby_order,
                    dodge=False if splitby is None else split,
                    orient="vertical",
                    fliersize=0,
                )
            else:
                raise ValueError("plot_type should be 'violin' or 'box'")
            # plot horizontal lines
            if hlines is not None:
                # ignore if current variable is not given
                if variable in hlines.keys():
                    _ax.axhline(hlines[variable], color="k", ls="--", lw=1.8)
            if strip:
                if points_colorby is None:
                    sns.stripplot(
                        x="group",
                        y="value",
                        data=tmp,
                        hue=splitby if splitby is not None else "group",
                        hue_order=(
                            splitby_order if splitby is not None else groupby_order
                        ),
                        dodge=False if splitby is None else True,
                        jitter=jitter,
                        color="k" if splitby is None else None,
                        palette=None if splitby is None else "dark:black",
                        size=size,
                        edgecolor="k",
                        linewidth=0.5,
                        alpha=0.7,
                        ax=_ax,
                    )
                else:
                    if points_colordict is None:
                        # get the Accent colormap by default
                        Accent = mcp.gen_color(
                            cmap="rainbow", n=len(tmp.points_hue.cat.categories)
                        )
                        points_colordict = dict(
                            zip(
                                tmp.points_hue.cat.categories,
                                Accent[: len(tmp.points_hue.cat.categories)],
                            )
                        )
                    else:
                        # check if given points_colordict contains all categories
                        if not all(
                            [
                                j in tmp.points_hue.cat.categories
                                for j in points_colordict.keys()
                            ]
                        ):
                            print(
                                "\tElements in `points_colordict` do not match `points_colorby` categories... using default accent colormap"
                            )
                            # get the Accent colormap instead
                            Accent = mcp.gen_color(
                                cmap="rainbow", n=len(tmp.points_hue.cat.categories)
                            )
                            points_colordict = dict(
                                zip(
                                    tmp.points_hue.cat.categories,
                                    Accent[: len(tmp.points_hue.cat.categories)],
                                )
                            )
                    # loop through colordict and plot points
                    for cat, color in points_colordict.items():
                        df_per_color = tmp.loc[tmp.points_hue == cat]
                        sns.stripplot(
                            data=df_per_color,
                            x="group",
                            y="value",
                            hue=splitby if splitby is not None else "group",
                            hue_order=(
                                splitby_order if splitby is not None else groupby_order
                            ),
                            dodge=False if splitby is None else True,
                            jitter=jitter,
                            palette=[color] * 2,
                            size=size,
                            edgecolor="k",
                            linewidth=0.5,
                            alpha=0.7,
                            ax=_ax,
                        )

            if sig:
                sig_tmp = calc_significance_split(
                    df=df,
                    ax=_ax,
                    feature=variable,
                    groupby="group",
                    splitby=splitby,
                    method=test_method,
                    transform=test_transform,
                    groupby_order=groupby_order,
                    splitby_order=splitby_order,
                    log_scale=log_scale,
                )
                sig_out = pd.concat([sig_out, pd.DataFrame(sig_tmp)])

            if log_scale is not None:
                assert log_scale_method in [
                    "functionlog",
                    "symlog",
                ], "Must provide 'functionlog' or 'symlog' for log_scale_method"
                if log_scale_method == "functionlog":
                    # use custom functions for transformation
                    _ax.set_yscale(
                        "functionlog",
                        functions=[
                            lambda x: x + pseudocount,
                            lambda x: x - pseudocount,
                        ],
                        base=log_scale,
                    )
                elif log_scale_method == "symlog":
                    _ax.set_yscale(
                        "symlog",
                        linthresh=pseudocount,
                        base=log_scale,
                    )
                # show log values as scalar (non-sci notation)
                _ax.yaxis.set_major_formatter(ScalarFormatter())

            # modify x-axis tick labels
            old_labels = [label.get_text() for label in _ax.get_xticklabels()]
            if sample_n:
                # modify x-axis tick labels to include sample size
                group_counts = tmp["group"].value_counts()
                new_labels = [
                    f"{label}\n(n = {group_counts[label]})" for label in old_labels
                ]
            else:
                new_labels = old_labels.copy()
            # set x-tick labels with conditional rotation
            for label, old_label in zip(_ax.get_xticklabels(), old_labels):
                if xlab_rot is None:
                    if len(old_label) > 10:
                        label.set_rotation(90)
                    else:
                        label.set_rotation(0)
                    label.set_horizontalalignment("center")
            # set label text
            if xlab_rot is None:
                _ax.set_xticklabels(new_labels)
            else:
                _ax.set_xticklabels(
                    new_labels,
                    rotation=xlab_rot,
                    ha="right" if xlab_rot != 0 else "center",
                    rotation_mode="anchor",
                )

            _ax.set_xlabel("")
            _ax.set_title(variable if titles is None else titles[iv])
            if _ax.get_legend() is not None:
                _ax.get_legend().remove()
            _ax.set_ylabel(variable if ylabel is None else ylabel)

        if legend:
            # create legend for outside plots
            legend_elements = None
            if splitby is not None:
                # get the Tab10 colormap
                tab10 = matplotlib.colormaps["tab10"]
                if splitby_order is None:
                    # get unique values in 'hue' column, using tab10 to match violin
                    legend_elements = [
                        Patch(
                            facecolor=tab10.colors[i],
                            edgecolor="k",
                            linewidth=0.5,
                            label=df_tidy.hue.unique()[i],
                        )
                        for i in range(len(df_tidy.hue.unique()))
                    ]
                else:
                    # get unique values in 'hue' column, using tab10 to match violin
                    legend_elements = [
                        Patch(
                            facecolor=tab10.colors[i],
                            edgecolor="k",
                            linewidth=0.5,
                            label=splitby_order[i],
                        )
                        for i in range(len(splitby_order))
                    ]
            if points_colorby is not None:
                points = [
                    Line2D(
                        [0],
                        [0],
                        label=cat,
                        marker="o",
                        markersize=10,
                        markerfacecolor=color,
                        markeredgewidth=0.5,
                        markeredgecolor="k",
                        alpha=0.7,
                        linestyle="None",
                    )
                    for cat, color in points_colordict.items()
                ]
                legend_elements = (
                    points if splitby is None else legend_elements + points
                )
            if legend_elements:
                plt.legend(
                    handles=legend_elements,
                    loc="upper left",
                    bbox_to_anchor=(1, 0.95),
                    frameon=False,
                )

        gs.tight_layout(fig)

    # return figure and/or stats
    if save is None:
        if sig:
            return fig, sig_out
        else:
            return fig
    else:
        print("Saving to {}".format(save))
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        if sig:
            return sig_out


def boxplots_group(
    a,
    features,
    groupby,
    groupby_order=None,
    groupby_colordict=None,
    points_colorby=None,
    points_colordict=None,
    pairby=None,
    layer=None,
    log_scale=None,
    log_scale_method="functionlog",
    pseudocount=1.0,
    sig=True,
    test_method="t-test",
    test_transform=None,
    test_pairs=None,
    bonferroni=False,
    sample_n=False,
    ylabel=None,
    xlab_rot=None,
    titles=None,
    legend=True,
    size=3,
    hlines=None,
    panelsize=(3, 3),
    ncols=6,
    outdir="./",
    save_prefix="",
    dpi=300,
):
    """
    Plot trends from  `a.obs` metadata. Save all plots in
    grid of single `.png` file.

    Parameters
    ----------
    a : Union[anndata.Anndata, pd.DataFrame]
        The annotated data matrix of shape `n_obs` by `n_vars`. Rows correspond to
        samples and columns to genes. Can also be `pd.DataFrame`.
    features : list of str
        List of genes, `.obs` columns, or DataFrame columns to plot (if `a` is
        `pd.DataFrame`) (y variable)
    groupby : list of str
        Columns from `a` or `a.obs` to group by (x variable)
    groupby_order : list of str, optional (default=`None`)
        List of values in `a[groupby]` or `a.obs[groupby]` specifying the order of
        groups on x-axis. If `groupby` is a list, `groupby_order` should also be a list
        with corresponding orders in each element.
    groupby_colordict : dict, optional (default=`None`)
        Dictionary of group, color pairs from `groupby` to color boxes and points by
    points_colorby : str, optional (default=`None`)
        Categorical `.obs` column to color stripplot points by.
    points_colordict : dict, optional (default=`None`)
        Dictionary of group, color pairs from `points_colorby` to color points by
    pairby : str, optional (default=`None`)
        Categorical `.obs` column identifying point pairings to draw lines between
        across `groupby` categories.
    layer : str, optional (default=`None`)
        Key from `layers` attribute of `adata` if present
    log_scale : int, optional (default=`None`)
        Set axis scale(s) to log. Numeric values are interpreted as the desired base
        (e.g. 10). When `None`, plot defers to the existing Axes scale.
    log_scale_method : literal ['functionlog','symlog'], optional (default='functionlog')
        Method for adjusting y-axis scale if `log_scale` is not `None`
    pseudocount : float, optional (default=1.0)
        Pseudocount to add to values before log-transforming with base=`log_scale`
    sig : bool, optional (default=`False`)
        Perform significance testing (2-way t-test) between all groups and add
        significance bars to plot(s)
    test_method : literal ["t-test","wilcoxon"], optional (default="t-test")
        Method for performing significance testing across groups
    test_transform : literal ["log1p","log10p","log2p","arcsinh"], optional (default=`None`)
        Transformation to apply to values before testing. "log1p"=np.log(val+1),
        "log10p"=np.log10(val+1), "log2p"=np.log2(val+1), "arcsinh"=np.arcsinh(val).
    test_pairs : list of tuple, optional (default=`None`)
        List of pairs of `groupby` values to include in testing. If `None`, test all
        levels in `groupby` pairwise.
    bonferroni : bool, optional (default=False)
        Adjust significance p-values with simple Bonferroni correction
    sample_n : bool, optional (default=False)
        Add sample number to x-axis tick labels for each group
    ylabel : str, optional (default=`None`)
        Label for y axes. If `None` use `colors`.
    xlab_rot : int, optional (default=`None`)
        Rotation (in degrees) to force on all x ticklabels. If `None`, auto-rotate x
        ticklabels 90 degrees if they're longer than 10 characters.
    titles : list of str, optional (default=`None`)
        Titles for each set of axes. If `None` use `features`.
    legend : bool, optional (default=`True`)
        Add legend to plot
    size : int, optional (default=3)
        Size of the jitter points
    hlines : dict, optional (default=`None`)
        Dictionary containing values to draw horizontal lines at for each feature.
        e.g. `{"feature1":10,"feature2":12,"feature3":8}` where "feature1", etc. are
        found in `features`.
    panelsize : tuple of float, optional (default=(3, 3))
        Size of each panel in output figure in inches
    ncols : int, optional (default=5)
        Number of columns in gridspec
    outdir : str, optional (default="./")
        Path to output directory for saving plots
    save_prefix : str, optional (default="")
        Prefix to add to filenames for saving
    dpi : float, optional (default=300)
        Resolution in dots per inch for saving figure. Ignored if `save_prefix` is
        `None`.

    Returns
    -------
    gs : gridspec.GridSpec
        Return gridspec object if `save==None`. Otherwise, write to `.png` in
        `outdir/`.
    sig_out : dict
        Dictionary of t-test statistics if `sig==True`. Otherwise, write to `.csv` in
        `outdir/`.
    """
    # coerce single string to list for looping
    if isinstance(features, str):
        features = [features]
    if isinstance(groupby, str):
        # coerce single string to list for looping
        groupby = [groupby]

    # prep df for plotting
    if isinstance(a, AnnData):
        # get genes of interest
        extra_cols = groupby.copy()
        if pairby is not None:
            extra_cols.append(pairby)
        if points_colorby is not None:
            extra_cols.append(points_colorby)
        # remove repetitive cols
        extra_cols = [x for x in list(set(extra_cols)) if x not in features]
        df = obs_df(
            a,
            features + extra_cols if len(extra_cols) > 0 else features,
            layer=layer,
        )
        new_features = [x for x in features if x not in list(set(extra_cols))]
    elif isinstance(a, pd.DataFrame):
        df = a.copy()
        new_features = features
    else:
        raise ValueError("'a' must be AnnData object or pandas DataFrame")

    # groupby_order if desired
    if groupby_order is not None:
        # if it's a single list, coerce to list of list
        if all(isinstance(g, str) for g in groupby_order):
            groupby_order = [groupby_order]
        else:
            assert all(
                isinstance(g, list) for g in groupby_order
            ), "'groupby_order' must contain only lists if len(groupby) > 1"

    # make boxplots for each `groupby`
    for ix, x in enumerate(groupby):
        print("Generating boxplots for {}".format(x))

        # set categories
        df[x] = df[x].astype("category")

        # groupby_order
        if groupby_order is not None:
            # check if given groupby_order contains all categories
            if all([j in df[x].cat.categories for j in groupby_order[ix]]):
                # reorder categories with manual groupby_order
                df[x] = df[x].cat.set_categories(groupby_order[ix], ordered=True)
            else:
                # ignore groupby_order due to mismatch
                print(
                    "\tElements in `groupby_order` do not match `groupby` categories... ignoring manual ordering"
                )
                # set order using cat.categories for downstream
                groupby_order[ix] = list(df[x].cat.categories)

        # seaborn plot style
        with sns.axes_style("whitegrid"):
            # generate gs object
            gs, fig = build_gridspec(
                panels=new_features, ncols=ncols, panelsize=panelsize
            )
            if sig:
                # initialize df of p values
                sig_out = pd.DataFrame(
                    {
                        "variable": [],
                        "group1": [],
                        "group2": [],
                        "pvalue": [],
                        "pvalue_adj": [],
                    }
                )
            for ic, c in enumerate(new_features):
                _ax = plt.subplot(gs[ic])
                sns.boxplot(
                    data=df,
                    x=x,
                    y=c,
                    hue=x,
                    palette=(
                        groupby_colordict if groupby_colordict is not None else None
                    ),
                    dodge=False,
                    saturation=0.4,
                    fliersize=0,
                    ax=_ax,
                )
                # plot horizontal lines
                if hlines is not None:
                    # ignore if current variable is not given
                    if c in hlines.keys():
                        _ax.axhline(hlines[c], color="k", ls="--", lw=1.8)
                if points_colorby is None:
                    sns.stripplot(
                        data=df,
                        x=x,
                        y=c,
                        hue=x,
                        palette=(
                            groupby_colordict if groupby_colordict is not None else None
                        ),
                        jitter=True,
                        dodge=False,
                        size=size,
                        edgecolor="k",
                        linewidth=0.5,
                        alpha=0.7,
                        ax=_ax,
                    )
                else:
                    # coerce points_colorby column to categorical
                    if not pd.api.types.is_categorical_dtype(df[points_colorby]):
                        df[points_colorby] = df[points_colorby].astype("category")
                    if points_colordict is None:
                        # get the Accent colormap by default
                        Accent = mcp.gen_color(
                            cmap="rainbow", n=len(df[points_colorby].cat.categories)
                        )
                        points_colordict = dict(
                            zip(
                                df[points_colorby].cat.categories,
                                Accent[: len(df[points_colorby].cat.categories)],
                            )
                        )
                    else:
                        # check if given points_colordict contains all categories
                        if not all(
                            [
                                j in df[points_colorby].cat.categories
                                for j in points_colordict.keys()
                            ]
                        ):
                            print(
                                "\tElements in `points_colordict` do not match `points_colorby` categories... using default accent colormap"
                            )
                            # get the Accent colormap instead
                            Accent = mcp.gen_color(
                                cmap="rainbow", n=len(df[points_colorby].cat.categories)
                            )
                            points_colordict = dict(
                                zip(
                                    df[points_colorby].cat.categories,
                                    Accent[: len(df[points_colorby].cat.categories)],
                                )
                            )
                    # loop through colordict and plot points
                    for cat, color in points_colordict.items():
                        df_per_color = df.loc[df[points_colorby] == cat]
                        sns.stripplot(
                            data=df_per_color,
                            x=x,
                            y=c,
                            hue=x,
                            jitter=True,
                            dodge=False,
                            palette=[color] * 2,
                            size=size,
                            edgecolor="k",
                            linewidth=0.5,
                            alpha=0.7,
                            ax=_ax,
                        )
                # add lines connecting paired points
                if pairby is not None:
                    pairs = df[pairby].unique()
                    for pair in pairs:
                        pair_data = df[df[pairby] == pair]
                        if len(pair_data) > 1:
                            # ensure the order of x-axis categories
                            if groupby_order is not None:
                                ordered_pair_data = []
                                for category in groupby_order[ix]:
                                    ordered_pair_data.append(
                                        pair_data[pair_data[x] == category]
                                    )
                                pair_data = pd.concat(ordered_pair_data)
                            else:
                                ordered_pair_data = []
                                for category in df[x].cat.categories:
                                    ordered_pair_data.append(
                                        pair_data[pair_data[x] == category]
                                    )
                                pair_data = pd.concat(ordered_pair_data)

                            for i in range(len(pair_data)):
                                for j in range(i + 1, len(pair_data)):
                                    # only plot across categories
                                    if pair_data.iloc[i][x] != pair_data.iloc[j][x]:
                                        _ax.plot(
                                            [
                                                pair_data.iloc[i][x],
                                                pair_data.iloc[j][x],
                                            ],
                                            [
                                                pair_data.iloc[i][c],
                                                pair_data.iloc[j][c],
                                            ],
                                            color="gray",
                                            linestyle="-",
                                            linewidth=0.8,
                                            alpha=0.8,
                                            label="_nolegend_",  # exclude from legend
                                        )

                if sig:
                    sig_tmp = calc_significance(
                        df=df,
                        ax=_ax,
                        feature=c,
                        groupby=x,
                        groupby_order=(
                            groupby_order[ix]
                            if groupby_order is not None
                            else df[x].cat.categories
                        ),
                        test_pairs=test_pairs,
                        method=test_method,
                        transform=test_transform,
                        bonferroni=bonferroni,
                        log_scale=log_scale,
                    )
                    sig_out = pd.concat([sig_out, pd.DataFrame(sig_tmp)])

                if log_scale is not None:
                    assert log_scale_method in [
                        "functionlog",
                        "symlog",
                    ], "Must provide 'functionlog' or 'symlog' for log_scale_method"
                    if log_scale_method == "functionlog":
                        # use custom functions for transformation
                        _ax.set_yscale(
                            "functionlog",
                            functions=[
                                lambda x: x + pseudocount,
                                lambda x: x - pseudocount,
                            ],
                            base=log_scale,
                        )
                    elif log_scale_method == "symlog":
                        _ax.set_yscale(
                            "symlog",
                            linthresh=pseudocount,
                            base=log_scale,
                        )
                    # show log values as scalar (non-sci notation)
                    _ax.yaxis.set_major_formatter(ScalarFormatter())

                # modify x-axis tick labels
                old_labels = [label.get_text() for label in _ax.get_xticklabels()]
                if sample_n:
                    # modify x-axis tick labels to include sample size
                    group_counts = df[x].value_counts()
                    new_labels = [
                        f"{label}\n(n = {group_counts[label]})" for label in old_labels
                    ]
                else:
                    new_labels = old_labels.copy()
                # set x-tick labels with conditional rotation
                for label, old_label in zip(_ax.get_xticklabels(), old_labels):
                    if xlab_rot is None:
                        if len(old_label) > 10:
                            label.set_rotation(90)
                        else:
                            label.set_rotation(0)
                        label.set_horizontalalignment("center")
                # set label text
                if xlab_rot is None:
                    _ax.set_xticklabels(new_labels)
                else:
                    _ax.set_xticklabels(
                        new_labels,
                        rotation=xlab_rot,
                        ha="right" if xlab_rot != 0 else "center",
                        rotation_mode="anchor",
                    )

                _ax.set_title(c if titles is None else titles[ic])
                _ax.set_xlabel("")
                _ax.set_ylabel(c if ylabel is None else ylabel)

            if legend:
                # create legend for outside plots
                if points_colorby is not None:
                    legend_elements = [
                        Line2D(
                            [0],
                            [0],
                            label=cat,
                            marker="o",
                            markersize=10,
                            markerfacecolor=color,
                            markeredgewidth=1,
                            markeredgecolor="k",
                            linestyle="None",
                        )
                        for cat, color in points_colordict.items()
                    ]
                else:
                    legend_elements = None
                if legend_elements:
                    plt.legend(
                        handles=legend_elements,
                        loc="upper left",
                        bbox_to_anchor=(1, 0.95),
                        frameon=False,
                    )

            gs.tight_layout(fig)

        # return figure and/or stats
        if save_prefix is None:
            if sig:
                return gs, sig_out
            else:
                return gs
        # save figure and/or stats
        else:
            if sig:
                print(
                    "Saving plot to {}".format(
                        os.path.join(
                            outdir, "{}{}_boxplots_sig.png".format(save_prefix, x)
                        )
                    )
                )
                fig.savefig(
                    os.path.abspath(
                        os.path.join(
                            outdir, "{}{}_boxplots_sig.png".format(save_prefix, x)
                        )
                    ),
                    dpi=dpi,
                    bbox_inches="tight",
                )
                # save statistics to .csv file
                print(
                    "Saving stats to {}".format(
                        os.path.join(
                            outdir,
                            "{}{}_boxplots_pvals.csv".format(save_prefix, x),
                        )
                    )
                )
                pd.DataFrame(sig_out).to_csv(
                    os.path.abspath(
                        os.path.join(
                            outdir,
                            "{}{}_boxplots_pvals.csv".format(save_prefix, x),
                        )
                    ),
                    index=False,
                )
            else:
                print(
                    "Saving plot to {}".format(
                        os.path.join(outdir, "{}{}_boxplots.png".format(save_prefix, x))
                    )
                )
                fig.savefig(
                    os.path.abspath(
                        os.path.join(outdir, "{}{}_boxplots.png".format(save_prefix, x))
                    ),
                    dpi=dpi,
                    bbox_inches="tight",
                )


def jointgrid_boxplots_category(
    a,
    x,
    y,
    color,
    figheight=5,
    sig=True,
    bonferroni=False,
    cmap_dict=None,
    stripplot=True,
    outdir="./",
    save_prefix=None,
    dpi=300,
):
    """
    Jointplot with scatter between two variables and marginal boxplots showing
    distributions and stats across a third variable (color)

    Parameters
    ----------
    a : Union[anndata.Anndata, pd.DataFrame]
        The annotated data matrix of shape `n_obs` by `n_vars`. Rows correspond to
        samples and columns to genes. Can also be `pd.DataFrame`.
    x : str
        Column from `a` or `a.obs` to plot on x axis of jointgrid
    y : str
        Column from `a` or `a.obs` to plot on y axis of jointgrid
    color : str
        Column from `a` or `a.obs` containing categories for marginal boxplots and
        statistics in `x` and `y`.
    figheight : float, optional (default=5)
        Size of output figure in inches (it will be square)
    sig : bool, optional (default=True)
        Perform significance testing (2-way t-test) between all groups and add
        significance bars to marginal boxplots
    bonferroni : bool, optional (default=False)
        Adjust significance p-values with simple Bonferroni correction
    cmap_dict : dictionary, optional (default=None)
        Dictionary of group, color pairs from `color` to color boxes and points by
    stripplot : bool, optional (default=`True`)
        Plot stripplot with jittered points over marginal boxplots
    outdir : str
        Path to output directory for saving plots
    save_prefix : str, optional (default=`None`)
        Prefix to add to filenames for saving. If `None`, don't save anything.
    dpi : float, optional (default=300)
        Resolution in dots per inch for saving figure. Ignored if `save_prefix` is
        `None`.

    Returns
    -------
    If `save_prefix!=None`, saves plot as `.png` file to `outdir` and stats (if
    `sig==True`) to `.csv` file.
    g : sns.JointGrid
        Plot object
    sig_out : dict
        Dictionary containing statistics (if `sig==True`)
    """
    # extract obs df if a is AnnData object
    if isinstance(a, AnnData):
        df = a.obs.copy()
    # otherwise, treat a as pd.DataFrame
    else:
        df = a.copy()
    # initialize jointgrid with clean plot style
    with sns.axes_style("white"):
        g = sns.JointGrid(
            data=df,
            x=x,
            y=y,
            hue=color,
            space=0,
            ratio=3,
            palette=cmap_dict,
            height=figheight,
            marginal_ticks=True,
        )
        g.plot_joint(sns.scatterplot)  # scatter in primary axes
        # marginal boxplots first
        sns.boxplot(
            df,
            x=g.hue,
            y=g.y,
            ax=g.ax_marg_y,
            fliersize=0 if stripplot else 3,
            palette=cmap_dict,
            legend=False,
        )
        sns.boxplot(
            df,
            y=g.hue,
            x=g.x,
            ax=g.ax_marg_x,
            fliersize=0 if stripplot else 3,
            palette=cmap_dict,
            legend=False,
        )
        # make boxplots black and white
        for ax in [g.ax_marg_x, g.ax_marg_y]:
            plt.setp(ax.lines, color="k")
            for i, box in enumerate(ax.patches):
                plt.setp(box, edgecolor="k")

        if stripplot:
            # marginal stripplots over boxes
            sns.stripplot(
                df,
                x=g.hue,
                y=g.y,
                ax=g.ax_marg_y,
                hue=g.hue,
                legend=None,
                palette=cmap_dict,
                size=3,
                edgecolor="k",
                linewidth=0.5,
                alpha=0.7,
            )
            sns.stripplot(
                df,
                y=g.hue,
                x=g.x,
                ax=g.ax_marg_x,
                hue=g.hue,
                legend=None,
                palette=cmap_dict,
                size=3,
                edgecolor="k",
                linewidth=0.5,
                alpha=0.7,
            )
        # compile statistics across `color` for `x` and `y`
        if sig:
            # initialize dictionary of p values
            sig_out = {
                "variable": [],
                "group1": [],
                "group2": [],
                "pvalue": [],
                "pvalue_adj": [],
            }
            # correlation (Pearson)
            corr_coef, corr_p_value = stats.pearsonr(df[x], df[y])
            sig_out["variable"].append("Pearson")
            sig_out["group1"].append(x)
            sig_out["group2"].append(y)
            sig_out["pvalue"].append(corr_p_value)
            sig_out["pvalue_adj"].append(np.nan)
            g.ax_joint.annotate(
                "$R^2$: {}".format(np.round(corr_coef, 3)),
                xy=(0.05, 0.95),
                xycoords="axes fraction",
            )

            sig_out = pd.DataFrame(sig_out)

            # boxplot marginal y axis
            sig_tmp = calc_significance(
                df=df,
                ax=g.ax_marg_y,
                feature=y,
                groupby=color,
                groupby_order=None,
                test_pairs=None,
                bonferroni=bonferroni,
                log_scale=None,
            )
            sig_out = pd.concat([sig_out, pd.DataFrame(sig_tmp)])
            # boxplot marginal x axis
            sig_tmp = calc_significance(
                df=df,
                ax=g.ax_marg_x,
                feature=x,
                groupby=color,
                groupby_order=None,
                test_pairs=None,
                bonferroni=bonferroni,
                log_scale=None,
                horizontal=True,  # draw sig bars horizontally
            )
            sig_out = pd.concat([sig_out, pd.DataFrame(sig_tmp)])

        # set x labels for marginal boxplot
        g.ax_marg_y.set_xticklabels(
            g.ax_marg_y.get_xticklabels(), rotation=45, ha="right"
        )
        # remove x label for marginal boxplot
        g.ax_marg_y.set_xlabel("")
        # remove y label for marginal boxplot
        g.ax_marg_x.set_ylabel("")

        # despine marginal boxplots
        sns.despine(ax=g.ax_marg_x, left=True)
        sns.despine(ax=g.ax_marg_y, bottom=True)

        # return figure and/or stats
        if save_prefix is None:
            if sig:
                return g, sig_out
            else:
                return g
        # save figure and/or stats
        else:
            if sig:
                g.savefig(
                    os.path.abspath(
                        os.path.join(
                            outdir,
                            "{}_jointgrid_boxplots_cat_sig.png".format(save_prefix),
                        )
                    ),
                    dpi=dpi,
                    bbox_inches="tight",
                )
                # save statistics to .csv file
                pd.DataFrame(sig_out).to_csv(
                    os.path.abspath(
                        os.path.join(
                            outdir,
                            "{}_jointgrid_boxplots_cat_pvals.csv".format(save_prefix),
                        )
                    ),
                    index=False,
                )
                return g, sig_out
            else:
                g.savefig(
                    os.path.abspath(
                        os.path.join(
                            outdir, "{}_jointgrid_boxplots_cat.png".format(save_prefix)
                        )
                    ),
                    dpi=dpi,
                    bbox_inches="tight",
                )
                return g


def jointgrid_boxplots_threshold(
    a,
    x,
    y,
    color,
    x_thresh,
    figheight=5,
    sig=True,
    cmap_dict=None,
    stripplot=True,
    dodge_by_color=False,
    outdir="./",
    save_prefix=None,
    dpi=300,
):
    """
    Jointplot with scatter between two variables and marginal boxplots showing
    distributions and stats across a third variable (color, x values in y margin)
    and a threshold of x (x_thresh, y values in x margin)

    Parameters
    ----------
    a : Union[anndata.Anndata, pd.DataFrame]
        The annotated data matrix of shape `n_obs` by `n_vars`. Rows correspond to
        samples and columns to genes. Can also be `pd.DataFrame`.
    x : str
        Column from `a` or `a.obs` to plot on x axis of jointgrid
    y : str
        Column from `a` or `a.obs` to plot on y axis of jointgrid
    color : str
        Column from `a` or `a.obs` containing categories for marginal boxplots and
        statistics in `x` and `y`.
    x_thresh : float
        Threshold along `x` for which to split points by and plot marginal boxplots
        for `y`
    figheight : float, optional (default=5)
        Size of output figure in inches (it will be square)
    sig : bool, optional (default=True)
        Perform significance testing (2-way t-test) between all groups and add
        significance bars to marginal boxplots
    cmap_dict : dictionary, optional (default=None)
        Dictionary of group, color pairs from `color` to color boxes and points by
    stripplot : bool, optional (default=`True`)
        Plot stripplot with jittered points over marginal boxplots
    dodge_by_color : bool, optional (default=`False`)
        Dodge boxplots and stripplots in y-marginal axes by `color`. If `False`, only
        one boxplot per category ("-lo" and "-hi" based on `x_thresh`), with jitterplot
        points colored by `color` if `stripplot==True`.
    outdir : str
        Path to output directory for saving plots
    save_prefix : str, optional (default=`None`)
        Prefix to add to filenames for saving. If `None`, don't save anything.
    dpi : float, optional (default=300)
        Resolution in dots per inch for saving figure. Ignored if `save_prefix` is
        `None`.

    Returns
    -------
    If `save_prefix!=None`, saves plot as `.png` file to `outdir` and stats (if
    `sig==True`) to `.csv` file.
    g : sns.JointGrid
        Plot object
    sig_out : dict
        Dictionary containing statistics (if `sig==True`)
    """
    # extract obs df if a is AnnData object
    if isinstance(a, AnnData):
        df = a.obs.copy()
    # otherwise, treat a as pd.DataFrame
    else:
        df = a.copy()
    # threshold on x
    df["x_thresh_status"] = np.nan
    df.loc[df[x] >= x_thresh, "x_thresh_status"] = "{}-hi".format(x)
    df.loc[df[x] < x_thresh, "x_thresh_status"] = "{}-lo".format(x)
    # initialize jointgrid with clean plot style
    with sns.axes_style("white"):
        g = sns.JointGrid(
            data=df,
            x=x,
            y=y,
            hue=color,
            space=0,
            ratio=3,
            palette=cmap_dict,
            height=figheight,
            marginal_ticks=True,
        )
        g.plot_joint(sns.scatterplot)  # scatter in primary axes
        # marginal boxplots first
        sns.boxplot(
            df,
            x="x_thresh_status",
            y=g.y,
            hue=g.hue if dodge_by_color else None,
            ax=g.ax_marg_y,
            fliersize=0 if stripplot else 3,
            order=["{}-lo".format(x), "{}-hi".format(x)],
            color=None if dodge_by_color else "w",
            dodge=True if dodge_by_color else False,
            legend=False,
        )
        sns.boxplot(
            df,
            y=g.hue,
            x=g.x,
            ax=g.ax_marg_x,
            fliersize=0 if stripplot else 3,
            palette=cmap_dict,
            legend=False,
        )
        # make boxplots black and white
        for ax in [g.ax_marg_x, g.ax_marg_y]:
            plt.setp(ax.lines, color="k")
            for i, box in enumerate(ax.patches):
                plt.setp(box, edgecolor="k")

        if stripplot:
            # marginal stripplots over boxes
            sns.stripplot(
                df,
                x="x_thresh_status",
                y=g.y,
                hue=g.hue,
                ax=g.ax_marg_y,
                legend=None,
                palette=cmap_dict,
                order=["{}-lo".format(x), "{}-hi".format(x)],
                dodge=True if dodge_by_color else False,
                size=3,
                edgecolor="k",
                linewidth=0.5,
                alpha=0.7,
            )
            sns.stripplot(
                df,
                y=g.hue,
                x=g.x,
                ax=g.ax_marg_x,
                hue=g.hue,
                legend=None,
                palette=cmap_dict,
                size=3,
                edgecolor="k",
                linewidth=0.5,
                alpha=0.7,
            )
        # compile statistics across `color` and `x_thresh_status` for `x` and `y`
        if sig:
            # initialize dictionary of p values
            sig_out = {
                "variable": [],
                "group1": [],
                "group2": [],
                "pvalue": [],
            }
            # correlation (Pearson)
            corr_coef, corr_p_value = stats.pearsonr(df[x], df[y])
            sig_out["variable"].append("Pearson")
            sig_out["group1"].append(x)
            sig_out["group2"].append(y)
            sig_out["pvalue"].append(corr_p_value)
            g.ax_joint.annotate(
                "$R^2$: {}".format(np.round(corr_coef, 3)),
                xy=(0.05, 0.95),
                xycoords="axes fraction",
            )
            # y axis first
            sig_count = 0  # initiate significant count for bar height
            # by color within hi and lo groups
            if dodge_by_color:
                for i_sig_y in range(len(df[color].unique())):
                    for i_sig_y_2 in [
                        x
                        for x in range(i_sig_y, len(df[color].unique()))
                        if x != i_sig_y
                    ]:
                        # lo x_thresh_status
                        _, p_value = stats.ttest_ind(
                            df.loc[
                                (df["x_thresh_status"] == "{}-lo".format(x))
                                & (df[color] == df[color].unique()[i_sig_y]),
                                y,
                            ].dropna(),
                            df.loc[
                                (df["x_thresh_status"] == "{}-lo".format(x))
                                & (df[color] == df[color].unique()[i_sig_y_2]),
                                y,
                            ].dropna(),
                        )
                        # dump results into dictionary
                        sig_out["variable"].append(y)
                        sig_out["group1"].append(
                            "{} {}".format(
                                "{}-lo".format(x), df[color].unique()[i_sig_y]
                            )
                        )
                        sig_out["group2"].append(
                            "{} {}".format(
                                "{}-lo".format(x), df[color].unique()[i_sig_y_2]
                            )
                        )
                        sig_out["pvalue"].append(p_value)
                        # plot bars if significance
                        if p_value <= 0.05:
                            sig_count += 1  # increment significant count
                            if p_value < 0.0001:
                                displaystring = r"***"
                            elif p_value < 0.001:
                                displaystring = r"**"
                            else:
                                displaystring = r"*"
                            # offset by 10 percent for each significant pair
                            height = df[y].max() + 0.15 * df[y].max() * sig_count
                            # set up significance bar
                            bar_centers = np.array([i_sig_y - 0.25, i_sig_y_2 - 0.75])
                            significance_bar(
                                bar_centers[0],
                                bar_centers[1],
                                height,
                                displaystring,
                                ax=g.ax_marg_y,
                            )

                        # hi x_thresh_status
                        _, p_value = stats.ttest_ind(
                            df.loc[
                                (df["x_thresh_status"] == "{}-hi".format(x))
                                & (df[color] == df[color].unique()[i_sig_y]),
                                y,
                            ].dropna(),
                            df.loc[
                                (df["x_thresh_status"] == "{}-hi".format(x))
                                & (df[color] == df[color].unique()[i_sig_y_2]),
                                y,
                            ].dropna(),
                        )
                        # dump results into dictionary
                        sig_out["variable"].append(y)
                        sig_out["group1"].append(
                            "{} {}".format(
                                "{}-hi".format(x), df[color].unique()[i_sig_y]
                            )
                        )
                        sig_out["group2"].append(
                            "{} {}".format(
                                "{}-hi".format(x), df[color].unique()[i_sig_y_2]
                            )
                        )
                        sig_out["pvalue"].append(p_value)
                        # plot bars if significance
                        if p_value <= 0.05:
                            sig_count += 1  # increment significant count
                            # if lo was significant and hi is significant, decrease
                            # sig_count so lo and hi significance brackets are plotted
                            # side-by-side
                            if sig_count == 2:
                                sig_count = 1
                            if p_value < 0.0001:
                                displaystring = r"***"
                            elif p_value < 0.001:
                                displaystring = r"**"
                            else:
                                displaystring = r"*"
                            # offset by 10 percent for each significant pair
                            height = df[y].max() + 0.15 * df[y].max() * sig_count
                            # set up significance bar
                            bar_centers = np.array([i_sig_y + 0.75, i_sig_y_2 + 0.25])
                            significance_bar(
                                bar_centers[0],
                                bar_centers[1],
                                height,
                                displaystring,
                                ax=g.ax_marg_y,
                            )

            for i_sig_y in range(len(df["x_thresh_status"].unique())):
                for i_sig_y_2 in [
                    x
                    for x in range(i_sig_y, len(df["x_thresh_status"].unique()))
                    if x != i_sig_y
                ]:
                    # if label has more than two classes, automatically perform
                    # t-tests and add significance bars to plots
                    _, p_value = stats.ttest_ind(
                        df.loc[
                            df["x_thresh_status"]
                            == df["x_thresh_status"].unique()[i_sig_y],
                            y,
                        ].dropna(),
                        df.loc[
                            df["x_thresh_status"]
                            == df["x_thresh_status"].unique()[i_sig_y_2],
                            y,
                        ].dropna(),
                    )
                    # dump results into dictionary
                    sig_out["variable"].append(y)
                    sig_out["group1"].append(df["x_thresh_status"].unique()[i_sig_y])
                    sig_out["group2"].append(df["x_thresh_status"].unique()[i_sig_y_2])
                    sig_out["pvalue"].append(p_value)
                    # plot bars if significance
                    if p_value <= 0.05:
                        sig_count += 1  # increment significant count
                        if p_value < 0.0001:
                            displaystring = r"***"
                        elif p_value < 0.001:
                            displaystring = r"**"
                        else:
                            displaystring = r"*"
                        # offset by 10 percent for each significant pair
                        height = df[y].max() + 0.15 * df[y].max() * sig_count
                        # set up significance bar
                        bar_centers = np.array([i_sig_y, i_sig_y_2])
                        significance_bar(
                            bar_centers[0],
                            bar_centers[1],
                            height,
                            displaystring,
                            ax=g.ax_marg_y,
                        )

            # x axis next
            sig_count = 0  # initiate significant count for bar height
            for i_sig_x in range(len(df[color].unique())):
                for i_sig_x_2 in [
                    x for x in range(i_sig_x, len(df[color].unique())) if x != i_sig_x
                ]:
                    # if label has more than two classes, automatically perform
                    # t-tests and add significance bars to plots
                    _, p_value = stats.ttest_ind(
                        df.loc[df[color] == df[color].unique()[i_sig_x], x].dropna(),
                        df.loc[df[color] == df[color].unique()[i_sig_x_2], x].dropna(),
                    )
                    # dump results into dictionary
                    sig_out["variable"].append(x)
                    sig_out["group1"].append(df[color].unique()[i_sig_x])
                    sig_out["group2"].append(df[color].unique()[i_sig_x_2])
                    sig_out["pvalue"].append(p_value)
                    # plot bars if significance
                    if p_value <= 0.05:
                        sig_count += 1  # increment significant count
                        if p_value < 0.0001:
                            displaystring = r"***"
                        elif p_value < 0.001:
                            displaystring = r"**"
                        else:
                            displaystring = r"*"
                        # offset by 10 percent for each significant pair
                        height = df[x].max() + 0.15 * df[x].max() * sig_count
                        # set up significance bar
                        bar_centers = np.array([i_sig_x, i_sig_x_2])
                        significance_bar(
                            bar_centers[0],
                            bar_centers[1],
                            height,
                            displaystring,
                            ax=g.ax_marg_x,
                            horizontal=True,
                        )

        # draw a vertical line on the joint plot at x_thresh; also on the x margin plot
        for ax in (g.ax_joint, g.ax_marg_x):
            ax.axvline(x_thresh, color="k", ls="--", lw=2.0)
        # draw a vertical line between categories on the y marginal boxplot
        g.ax_marg_y.axvline(0.5, color="k", ls="--", lw=2.0, ymax=0.85)

        # set x labels for marginal boxplot
        g.ax_marg_y.set_xticklabels(
            ["{}-lo".format(x), "{}-hi".format(x)], rotation=45, ha="right"
        )
        # remove x label for marginal boxplot
        g.ax_marg_y.set_xlabel("")
        # remove y label for marginal boxplot
        g.ax_marg_x.set_ylabel("")

        # despine marginal boxplots
        sns.despine(ax=g.ax_marg_x, left=True)
        sns.despine(ax=g.ax_marg_y, bottom=True)

        # return figure and/or stats
        if save_prefix is None:
            if sig:
                return g, sig_out
            else:
                return g
        # save figure and/or stats
        else:
            if sig:
                g.savefig(
                    os.path.abspath(
                        os.path.join(
                            outdir,
                            "{}_jointgrid_boxplots_thresh_sig.png".format(save_prefix),
                        )
                    ),
                    dpi=dpi,
                    bbox_inches="tight",
                )
                # save statistics to .csv file
                pd.DataFrame(sig_out).to_csv(
                    os.path.abspath(
                        os.path.join(
                            outdir,
                            "{}_jointgrid_boxplots_thresh_pvals.csv".format(
                                save_prefix
                            ),
                        )
                    ),
                    index=False,
                )
                return g, sig_out
            else:
                g.savefig(
                    os.path.abspath(
                        os.path.join(
                            outdir,
                            "{}_jointgrid_boxplots_thresh.png".format(save_prefix),
                        )
                    ),
                    dpi=dpi,
                    bbox_inches="tight",
                )
                return g


def cluster_pie(
    adata,
    pie_by="batch",
    groupby="leiden",
    ncols=5,
    show=None,
    figsize=(5, 5),
):
    """
    Plots pie graphs showing makeup of cluster groups

    Parameters
    ----------

    adata : anndata.AnnData
        the data
    pie_by : str, optional (default="batch")
        adata.obs column to split pie charts by
    groupby : str, optional (default="leiden")
        adata.obs column to create pie charts for
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
    if adata.obs[groupby].value_counts().min() == 0:
        print(
            "Warning: unused categories detected in adata.obs['{}']; removing prior to building plots...".format(
                groupby
            )
        )
        adata.obs[groupby] = adata.obs[groupby].cat.remove_unused_categories()
    # get portions for each cluster
    pies = {}  # init empty dict
    for c in adata.obs[groupby].cat.categories:
        pies[c] = (
            adata.obs.loc[adata.obs[groupby] == c, pie_by].value_counts()
            / adata.obs[groupby].value_counts()[c]
        ).to_dict()
    n_panels = len(adata.obs[groupby].cat.categories)
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
    # get pie chart colors
    cdict = {}
    # use existing scanpy colors, if applicable
    if "{}_colors".format(pie_by) in adata.uns:
        for ic, c in enumerate(adata.obs[pie_by].cat.categories):
            cdict[c] = adata.uns["{}_colors".format(pie_by)][ic]
    else:
        cmap = plt.get_cmap("tab10")
        for ic, c in enumerate(adata.obs[pie_by].cat.categories):
            cdict[c] = cmap(np.linspace(0, 1, len(adata.obs[pie_by].cat.categories)))[
                ic
            ]
    for ipie, pie in enumerate(pies.keys()):
        plt.subplot(gs[ipie])
        plt.pie(
            pies[pie].values(),
            labels=pies[pie].keys(),
            colors=[cdict[x] for x in pies[pie].keys()],
            radius=0.85,
            wedgeprops=dict(width=0.5),
            textprops={"fontsize": 12},
        )
        plt.title(
            label="{}_{}".format(groupby, pie),
            loc="left",
            fontweight="bold",
            fontsize=16,
        )
    gs.tight_layout(fig)
    if show is False:
        return gs


def custom_heatmap(
    adata,
    groupby,
    features=None,
    vars_dict=None,
    cluster_obs=False,
    cluster_vars=False,
    groupby_order=None,
    groupby_colordict="native",
    layer=None,
    plot_type="dotplot",
    cmap="Greys",
    log_scale=None,
    linthresh=1.0,
    italicize_vars=True,
    colorbar_title="Mean expression\nin group",
    vars_axis="y",
    figsize=(5, 5),
    save=None,
    dpi=300,
    **kwargs,
):
    """
    Custom wrapper around `sc.pl.dotplot`, `sc.pl.matrixplot`, and
    `sc.pl.stacked_violin`

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object to plot from
    groupby : str
        Categorical column of `adata.obs` to group dotplot by
    features : list of str (default=`None`)
        List of features from `adata.obs.columns` or `adata.var_names` to plot. If
        `None`, then `vars_dict` must provided.
    vars_dict : dict, optional (default=`None`)
        Dictionary of groups of vars to highlight with brackets on dotplot. Keys are
        variable group names and values are list of variables found in `features`.
        If provided, `features` is ignored.
    cluster_obs : bool, optional (default=`False`)
        Hierarchically cluster `groupby` observations and show dendrogram
    cluster_vars : bool, optional (default=`False`)
        Hierarchically cluster `features` for a prettier dotplot. If `True`, return
        `features` in their new order (first return variable).
    groupby_order : list, optional (default=`None`)
        Explicit order for groups of observations from `adata.obs[groupby]`
    groupby_colordict : dict, str, or `None`, optional (default='native')
        Dictionary mapping `groupby` categories (keys) to colors (values) for the group
        axis text. If `groupby_colordict == 'native'`, use the colors available in
        `adata.uns[f'{groupby}_colors']`. If `None`, don't color group text on plot.
    layer : str
        Key from `adata.layers` to use for plotting gene values
    plot_type : str, optional (default="dotplot")
        One of "dotplot", "matrixplot", "dotmatrix", "stacked_violin", or "heatmap"
    cmap : str, optional (default="Greys")
        matplotlib colormap for dots
    log_scale : int, optional (default=`None`)
        Set axis scale(s) to symlog. Numeric values are interpreted as the desired base
        (e.g. 10). When `None`, plot defers to the existing Axes scale.
    linthresh : float, optional (default=1.0)
        The range within which the color scale is linear (to avoid having the plot go
        to infinity around zero).
    italicize_vars : bool, optional (default=`True`)
        Whether or not to italicize variable names on plot
    colorbar_title : str, optional (default="Mean expression\\nin group")
        Title for colorbar key
    vars_axis : str, optional (default="y")
        If "y", vars are shown on y-axis. If "x", vars are shown on x-axis.
    figsize : tuple of float, optional (default=(5,5))
        Size of the figure in inches
    save : str or `None`, optional (default=`None`)
        Path to file to save image to. If `None`, return figure object (second return
        variable if `cluster_vars == True`).
    dpi : float, optional (default=300)
        Resolution in dots per inch for saving figure. Ignored if `save` is `None`.
    **kwargs
        Keyword arguments to pass to `sc.pl.dotplot`, `sc.pl.matrixplot`,
        `sc.pl.stacked_violin`, or `sc.pl.heatmap`

    Returns
    -------
    features_ordered : list of str
        If `cluster_vars == True`, reordered `features` based on hierarchical
        clustering
    figure : sc._plotting object (DotPlot, matrixplot, stacked_violin)
        If `save == None`, scanpy plotting object is returned
    """
    # check that we're making a valid plot
    assert plot_type in [
        "dotplot",
        "matrixplot",
        "stacked_violin",
        "heatmap",
        "dotmatrix",
    ], "plot_type must be one of 'dotplot', 'matrixplot', 'dotmatrix', 'stacked_violin', 'heatmap'"
    # check for valid vars_axis
    assert vars_axis.lower() in [
        "x",
        "y",
    ], "Please provide either 'x' or 'y' for vars_axis"
    # get features from vars_dict if provided
    if vars_dict is not None:
        features = signature_dict_values(signatures_dict=vars_dict, unique=True)
    if np.all([x in adata.obs.columns for x in features]):
        same_origin = True
        print("Using {} features from adata.obs".format(len(features)))
        a_comb_sig = sc.AnnData(
            adata.obs[features].values,
            obs=adata.obs[[x for x in adata.obs.columns if x not in features]],
        )
        a_comb_sig.X = np.nan_to_num(a_comb_sig.X, 0)
        a_comb_sig.layers["raw_counts"] = a_comb_sig.X.copy()
        a_comb_sig.var_names = features
    elif np.all([x in adata.var_names for x in features]):
        same_origin = True
        print("Using {} features from adata.X".format(len(features)))
        a_comb_sig = adata[:, features].copy()
    else:
        same_origin = False
        print("Using {} features from adata.X and adata.obs".format(len(features)))
        a_comb_sig = adata[
            :, list(set(features).intersection(set(adata.var_names)))
        ].copy()

    # initialize dictionary of arguments
    args = {
        "adata": a_comb_sig,
        "groupby": groupby,
        "layer": layer,
        "swap_axes": True if vars_axis.lower() == "y" else False,
        "figsize": figsize,
        **kwargs,
    }

    # get colors for group names
    if groupby_colordict is not None:
        if isinstance(groupby_colordict, dict):
            args["adata"].uns["{}_colors".format(args["groupby"])] = [
                groupby_colordict[x]
                for x in args["adata"].obs[args["groupby"]].cat.categories
            ]
        elif groupby_colordict.lower() == "native":
            if "{}_colors".format(args["groupby"]) in adata.uns:
                group_colors = adata.uns["{}_colors".format(args["groupby"])]
                # add to .uns for heatmap plotting
                args["adata"].uns["{}_colors".format(args["groupby"])] = group_colors
                # extract as dictionary for axis label coloring
                groupby_colordict = dict(
                    zip(args["adata"].obs[args["groupby"]].cat.categories, group_colors)
                )
            else:
                print(
                    "{}_colors not found in adata.uns... skipping.".format(
                        args["groupby"]
                    )
                )
                groupby_colordict = None
        else:
            print("Invalid 'groupby_colordict' argument... skipping.")
            groupby_colordict = None

    if cluster_vars and vars_dict is None:
        assert (
            same_origin
        ), "In order to hierarchically cluster, features must all reside in .X or .obs"
        print("Hierarchically clustering features")
        # first get hierchically-clustered indices of variables
        link = linkage(a_comb_sig.X.T)
        leaves = leaves_list(link)
        # add to args dictionary
        args["var_names"] = a_comb_sig.var_names[leaves]  # use indices from clustermap

    elif vars_dict is not None:
        print("Using vars_dict for ordering features")
        # add to args dictionary
        args["var_names"] = vars_dict  # use variables dictionary

    else:
        print("Features ordered as given")
        # add to args dictionary
        args["var_names"] = features  # use variables as list

    if log_scale is not None:
        args["norm"] = SymLogNorm(base=log_scale, linthresh=linthresh)

    # create plot
    if plot_type in ["dotplot", "dotmatrix"]:
        args["dendrogram"] = cluster_obs if groupby_order is None else False
        args["categories_order"] = groupby_order if groupby_order is not None else None
        args["return_fig"] = True
        args["colorbar_title"] = colorbar_title
        myplot = sc.pl.dotplot(**args)
        # style options to plot
        if plot_type == "dotmatrix":
            myplot.style(
                cmap=cmap, dot_edge_color="k", dot_edge_lw=1, color_on="square"
            )
        elif plot_type == "dotplot":
            myplot.style(cmap=cmap, dot_edge_color="k", dot_edge_lw=1)
        # italicize variable names (genes)
        if italicize_vars and args["swap_axes"]:
            # alter y-axis labels
            myplot.get_axes()["mainplot_ax"].set_yticklabels(
                myplot.get_axes()["mainplot_ax"].get_yticklabels(), fontstyle="italic"
            )
        if italicize_vars and not args["swap_axes"]:
            # alter x-axis labels
            myplot.get_axes()["mainplot_ax"].set_xticklabels(
                myplot.get_axes()["mainplot_ax"].get_xticklabels(), fontstyle="italic"
            )
        # rotate size legend ticklabels
        sl = myplot.get_axes()["size_legend_ax"]
        sl.set_xticklabels(sl.get_xticklabels(), rotation=90)
        # rotate colorbar ticklabels
        cb = myplot.ax_dict["color_legend_ax"]
        # scalar format
        cb.xaxis.set_major_formatter(ScalarFormatter())
        cb.set_xticklabels(cb.get_xticklabels(), rotation=90)
    elif plot_type == "matrixplot":
        args["dendrogram"] = cluster_obs if groupby_order is None else False
        args["categories_order"] = groupby_order if groupby_order is not None else None
        args["return_fig"] = True
        args["colorbar_title"] = colorbar_title
        myplot = sc.pl.matrixplot(**args)
        # style options to plot
        myplot.style(cmap=cmap)
        # italicize variable names (genes)
        if italicize_vars and args["swap_axes"]:
            # alter y-axis labels
            myplot.get_axes()["mainplot_ax"].set_yticklabels(
                myplot.get_axes()["mainplot_ax"].get_yticklabels(), fontstyle="italic"
            )
        if italicize_vars and not args["swap_axes"]:
            # alter x-axis labels
            myplot.get_axes()["mainplot_ax"].set_xticklabels(
                myplot.get_axes()["mainplot_ax"].get_xticklabels(), fontstyle="italic"
            )
        # rotate colorbar ticklabels
        cb = myplot.get_axes()["color_legend_ax"]
        # scalar format
        cb.xaxis.set_major_formatter(ScalarFormatter())
        cb.set_xticklabels(cb.get_xticklabels(), rotation=90)
    elif plot_type == "stacked_violin":
        args["dendrogram"] = cluster_obs if groupby_order is None else False
        args["categories_order"] = groupby_order if groupby_order is not None else None
        args["return_fig"] = True
        args["colorbar_title"] = colorbar_title
        myplot = sc.pl.stacked_violin(**args)
        # style options to plot
        myplot.style(cmap=cmap, linewidth=1)
        # italicize variable names (genes)
        if italicize_vars and args["swap_axes"]:
            # alter y-axis labels
            myplot.get_axes()["mainplot_ax"].set_yticklabels(
                myplot.get_axes()["mainplot_ax"].get_yticklabels(), fontstyle="italic"
            )
        if italicize_vars and not args["swap_axes"]:
            # alter x-axis labels
            myplot.get_axes()["mainplot_ax"].set_xticklabels(
                myplot.get_axes()["mainplot_ax"].get_xticklabels(), fontstyle="italic"
            )
        # rotate colorbar ticklabels
        cb = myplot.get_axes()["color_legend_ax"]
        # scalar format
        cb.xaxis.set_major_formatter(ScalarFormatter())
        cb.set_xticklabels(cb.get_xticklabels(), rotation=90)
    elif plot_type == "heatmap":
        args["dendrogram"] = cluster_obs if groupby_order is None else False
        args["show"] = False  # don't show sc.pl.heatmap so we can manipulate plot obj
        args["cmap"] = cmap  # outside of .style for sc.pl.heatmap
        args["adata"].obs[args["groupby"]] = (
            args["adata"].obs[args["groupby"]].astype("category")
        )
        if groupby_order is not None:
            args["adata"].obs[args["groupby"]] = (
                args["adata"].obs[args["groupby"]].cat.reorder_categories(groupby_order)
            )
        myplot = sc.pl.heatmap(**args)
        # remove groupby axis label
        if args["swap_axes"]:
            myplot["groupby_ax"].set_xlabel("")
        else:
            myplot["groupby_ax"].set_ylabel("")
        # italicize variable names (genes)
        if italicize_vars and args["swap_axes"]:
            # alter y-axis labels
            myplot["heatmap_ax"].set_yticklabels(
                myplot["heatmap_ax"].get_yticklabels(), fontstyle="italic"
            )
        if italicize_vars and not args["swap_axes"]:
            # alter x-axis labels
            myplot["heatmap_ax"].set_xticklabels(
                myplot["heatmap_ax"].get_xticklabels(), fontstyle="italic"
            )
        # add label to colorbar ax
        cb = myplot["heatmap_ax"].images[-1].colorbar
        cb.set_label(colorbar_title)
        # scalar format
        cb.ax.yaxis.set_major_formatter(ScalarFormatter())

    # color group names according to groupby_colordict
    if groupby_colordict:
        if plot_type in [
            "dotplot",
            "matrixplot",
            "stacked_violin",
            "dotmatrix",
        ]:  # for 'dotplot' etc
            myplot.ax_dict = myplot.get_axes()
            if args["swap_axes"]:  # alter x axis
                myplot.ax_dict["mainplot_ax"].set_xticklabels(
                    myplot.ax_dict["mainplot_ax"].get_xticklabels(),
                    path_effects=[pe.withStroke(linewidth=0.2, foreground="k")],
                )
                [
                    t.set_color(i)
                    for (i, t) in zip(
                        [
                            groupby_colordict[x.get_text()]
                            for x in myplot.ax_dict["mainplot_ax"].get_xticklabels()
                        ],
                        myplot.ax_dict["mainplot_ax"].xaxis.get_ticklabels(),
                    )
                ]
            else:  # alter y axis
                myplot.ax_dict["mainplot_ax"].set_yticklabels(
                    myplot.ax_dict["mainplot_ax"].get_yticklabels(),
                    path_effects=[pe.withStroke(linewidth=0.2, foreground="k")],
                )
                [
                    t.set_color(i)
                    for (i, t) in zip(
                        [
                            groupby_colordict[y.get_text()]
                            for y in myplot.ax_dict["mainplot_ax"].get_yticklabels()
                        ],
                        myplot.ax_dict["mainplot_ax"].yaxis.get_ticklabels(),
                    )
                ]

        else:  # for 'heatmap'
            if args["swap_axes"]:  # alter x axis
                myplot["groupby_ax"].set_xticklabels(
                    myplot["groupby_ax"].get_xticklabels(),
                    path_effects=[pe.withStroke(linewidth=0.2, foreground="k")],
                )
                [
                    t.set_color(i)
                    for (i, t) in zip(
                        [
                            groupby_colordict[x.get_text()]
                            for x in myplot["groupby_ax"].get_xticklabels()
                        ],
                        myplot["groupby_ax"].xaxis.get_ticklabels(),
                    )
                ]
            else:  # alter y axis
                myplot["groupby_ax"].set_yticklabels(
                    myplot["groupby_ax"].get_yticklabels(),
                    path_effects=[pe.withStroke(linewidth=0.2, foreground="k")],
                )
                [
                    t.set_color(i)
                    for (i, t) in zip(
                        [
                            groupby_colordict[y.get_text()]
                            for y in myplot["groupby_ax"].get_yticklabels()
                        ],
                        myplot["groupby_ax"].yaxis.get_ticklabels(),
                    )
                ]

    # return figure and/or variable clustering
    if save is None:
        if cluster_vars:
            return a_comb_sig.var_names[leaves], myplot
        else:
            return myplot
    # save figure and return variable clustering
    else:
        print("Saving to {}".format(save))
        plt.savefig(save, dpi=dpi, bbox_inches="tight")
        if cluster_vars:
            return a_comb_sig.var_names[leaves]


def plot_embedding(
    adata,
    basis="X_umap",
    colors=None,
    show_clustering=True,
    ncols=5,
    n_cnmf_markers=7,
    figsize_scale=1.0,
    cmap="viridis",
    seed=18,
    save_to=None,
    verbose=True,
    **kwargs,
):
    """
    Plots reduced-dimension embeddings of single-cell dataset

    Parameters
    ----------
    adata : anndata.AnnData
        object containing preprocessed and dimension-reduced counts matrix
    basis : str, optional (default="X_umap")
        key from `adata.obsm` containing embedding coordinates
    colors : list of str, optional (default=None)
        colors to plot; can be genes or .obs columns
    show_clustering : bool, optional (default=True)
        plot PAGA graph and leiden clusters on first two axes
    basis : str, optional (default="X_umap")
        embedding to plot - key from `adata.obsm`
    ncols : int, optional (default=5)
        number of columns in gridspec
    n_cnmf_markers : int, optional (default=7)
        number of top genes to print on cNMF plots
    figsize_scale : float, optional (default=1.0)
        scaler for figure size. calculated using ncols to keep each panel square.
        values < 1.0 will compress figure, > 1.0 will expand.
    cmap : str, optional (default="viridis")
        valid color map for the plot
    seed : int, optional (default=18)
        random state for plotting PAGA
    save_to : str, optional (default=None)
        path to .png file for saving figure; default is plt.show()
    verbose : bool, optional (default=True)
        print updates to console
    **kwargs : optional
        args to pass to `sc.pl.embedding` (e.g. "size", "add_outline", etc.)

    Returns
    -------
    plot of embedding with overlays from "colors" as matplotlib gridspec object,
    unless `save_to` is not None.
    """
    if isinstance(colors, str):  # force colors into list if single string
        colors = [colors]
    if "paga" in adata.uns:
        cluster_colors = ["paga", "leiden"]
    else:
        cluster_colors = ["leiden"]
    if colors is not None:
        # get full list of things to plot on embedding
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
    fig = plt.figure(
        figsize=(ncols * n_cols * figsize_scale, ncols * n_rows * figsize_scale)
    )
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
                    fontsize="large",
                    fontoutline=2.0,
                    node_size_scale=2.5,
                    random_state=seed,
                )
                ax.set_title(label="PAGA", loc="left", fontweight="bold", fontsize=16)
            else:
                if color.lower() in [
                    "leiden",
                    "louvain",
                    "cluster",
                    "group",
                    "cell_type",
                ]:
                    leg_loc, leg_fontsize, leg_fontoutline = "on data", "large", 2.0
                else:
                    leg_loc, leg_fontsize, leg_fontoutline = (
                        "right margin",
                        12,
                        None,
                    )
                sc.pl.embedding(
                    adata,
                    basis=basis,
                    color=color,
                    ax=ax,
                    frameon=False,
                    show=False,
                    legend_loc=leg_loc,
                    legend_fontsize=leg_fontsize,
                    legend_fontoutline=leg_fontoutline,
                    title="",
                    color_map=cmap,
                    **kwargs,
                )
                # add top three gene loadings if cNMF
                if color.startswith("usage_"):
                    y_range = (
                        adata.obsm[basis][:, 1].max() - adata.obsm[basis][:, 1].min()
                    )
                    [
                        ax.text(
                            x=adata.obsm[basis][:, 0].max(),
                            y=adata.obsm[basis][:, 1].max() - (0.06 * y_range * x),
                            s=""
                            + adata.uns["cnmf_markers"].loc[x, color.split("_")[1]],
                            fontsize=12,
                            fontstyle="italic",
                            color="k",
                            ha="right",
                        )
                        for x in range(n_cnmf_markers)
                    ]
            if color in adata.var_names:  # italicize title if plotting a gene
                ax.set_title(
                    label=color,
                    loc="left",
                    fontweight="bold",
                    fontsize=16,
                    fontstyle="italic",
                )
            else:
                ax.set_title(label=color, loc="left", fontweight="bold", fontsize=16)
            unique_colors.remove(color)
            i = i + 1
    fig.tight_layout()
    if save_to is not None:
        if verbose:
            print("Saving embeddings to {}".format(save_to))
        plt.savefig(save_to)
    else:
        return fig


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
    if show is False:
        return gs


def rank_genes_groups(
    adata,
    groups=None,
    n_genes=20,
    gene_symbols=None,
    key="rank_genes_groups",
    fontsize=8,
    ncols=4,
    sharey=True,
    save_to=None,
    ax=None,
    include_lowest=False,
    panelsize=(4, 4),
    **kwds,
):
    """
    Plot ranking of genes with optional inclusion of lowest ranking genes.
    Adapted from sc.pl.rank_genes_groups

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groups : str | Sequence[str] | None, optional
        The groups for which to show the gene ranking.
    n_genes : int, optional
        Number of genes to show. Default is 20.
    gene_symbols : str | None, optional
        Key for field in `.var` that stores gene symbols if you do not want to
        use `.var_names`.
    key : str | None, optional
        Key for accessing the rank genes groups in `adata.uns`. Default is "rank_genes_groups".
    fontsize : int, optional
        Fontsize for gene names. Default is 8.
    ncols : int, optional
        Number of panels shown per row. Default is 4.
    sharey : bool, optional
        Controls if the y-axis of each panel should be shared. By passing
        `sharey=False`, each panel has its own y-axis range. Default is True.
    save_to : bool | None, optional
        Path to save the figure to. If None, the figure is not saved.
    ax : Axes | None, optional
        Matplotlib Axes object to draw the plots on. If None, a new figure is created.
    include_lowest : bool, optional
        Whether to include the lowest ranking genes. Default is False.
    panelsize : tuple, optional
        Tuple specifying the (width, height) of each panel. Default is (4, 4).
    **kwds
        Additional keyword arguments.

    Returns
    -------
    List of each group’s matplotlib axis or `None` if `show=True`.
    """
    n_panels_per_row = kwds.get("n_panels_per_row", ncols)
    if n_genes < 1:
        msg = (
            "Specifying a negative number for n_genes has not been implemented for "
            f"this plot. Received {n_genes=!r}."
        )
        raise NotImplementedError(msg)

    reference = str(adata.uns[key]["params"]["reference"])
    group_names = adata.uns[key]["names"].dtype.names if groups is None else groups
    n_panels_x = min(n_panels_per_row, len(group_names))
    n_panels_y = np.ceil(len(group_names) / n_panels_x).astype(int)

    from matplotlib import gridspec

    if ax is None or (sps := ax.get_subplotspec()) is None:
        fig = (
            plt.figure(
                figsize=(
                    n_panels_x * panelsize[0],
                    n_panels_y * panelsize[1],
                )
            )
            if ax is None
            else ax.get_figure()
        )
        gs = gridspec.GridSpec(n_panels_y, n_panels_x, fig, wspace=0.22, hspace=0.3)
    else:
        fig = ax.get_figure()
        gs = sps.subgridspec(n_panels_y, n_panels_x)
    if fig is None:
        msg = "passed ax has no associated figure"
        raise RuntimeError(msg)

    axs: list[Axes] = []
    ymin = np.inf
    ymax = -np.inf
    for count, group_name in enumerate(group_names):
        if include_lowest:
            gene_names = np.concatenate(
                (
                    adata.uns[key]["names"][group_name][:n_genes],
                    adata.uns[key]["names"][group_name][-n_genes:],
                )
            )
            scores = np.concatenate(
                (
                    adata.uns[key]["scores"][group_name][:n_genes],
                    adata.uns[key]["scores"][group_name][-n_genes:],
                )
            )
        else:
            gene_names = adata.uns[key]["names"][group_name][:n_genes]
            scores = adata.uns[key]["scores"][group_name][:n_genes]

        if sharey:
            ymin = min(ymin, np.min(scores))
            ymax = max(ymax, np.max(scores))

            axs.append(fig.add_subplot(gs[count], sharey=axs[0] if axs else None))
        else:
            ymin = np.min(scores)
            ymax = np.max(scores)
            ymax += 0.3 * (ymax - ymin)

            axs.append(fig.add_subplot(gs[count]))
            axs[-1].set_ylim(ymin, ymax)

        axs[-1].set_xlim(-0.9, len(gene_names) - 0.1)

        if gene_symbols is not None:
            if adata.raw is not None and adata.uns[key]["params"]["use_raw"]:
                gene_names = adata.raw.var[gene_symbols][gene_names]
            else:
                gene_names = adata.var[gene_symbols][gene_names]

        for ig, gene_name in enumerate(gene_names):
            axs[-1].text(
                ig,
                scores[ig],
                gene_name,
                rotation="vertical",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize=fontsize,
            )

        if include_lowest:
            axs[-1].axhline(0, linestyle="--", color="grey")

        axs[-1].set_title(f"{group_name} vs. {reference}")
        axs[-1].set_xticks([])  # remove x-ticks
        axs[-1].set_xticklabels([])  # remove x-tick labels

        if count >= n_panels_x * (n_panels_y - 1):
            axs[-1].set_xlabel("ranking")

        if count % n_panels_x == 0:
            axs[-1].set_ylabel("score")

    if sharey is True and axs:
        ymax += 0.3 * (ymax - ymin)
        axs[0].set_ylim(ymin, ymax)

    if save_to is None:
        return fig
    else:
        fig.tight_layout()
        fig.savefig(save_to)


def decoupler_dotplot(
    df,
    x,
    y,
    c,
    s,
    largest_dot=50,
    cmap="Reds",
    title=None,
    figsize=(3, 5),
    ax=None,
    return_fig=False,
    save=None,
    dpi=200,
):
    """
    Plot results of `decoupler` enrichment analysis as dots.

    Parameters
    ----------
    df : DataFrame
        Results of enrichment analysis.
    x : str
        Column name of `df` to use as continous value.
    y : str
        Column name of `df` to use as labels.
    c : str
        Column name of `df` to use for coloring.
    s : str
        Column name of `df` to use for dot size.
    largest_dot : int
        Parameter to control the size of the dots in points.
    cmap : str
        Colormap to use.
    title : str, None
        Text to write as title of the plot.
    figsize : tuple
        Figure size.
    ax : Axes, None
        A matplotlib axes object. If None returns new figure.
    return_fig : bool
        Whether to return a Figure object or not.
    save : str, None
        Path to where to save the plot. Infer the filetype if ending in
        {`.pdf`, `.png`, `.svg`}.
    dpi : float, optional (default=200)
        Resolution in dots per inch for saving figure. Ignored if `save` is `None`.

    Returns
    -------
    fig : matplotlib.Figure, None
        If `return_fig==True`, returns figure object.
    """
    # Extract from df
    x_vals = df[x].values
    if y is not None:
        y_vals = df[y].values
    else:
        y_vals = df.index.values
    c_vals = df[c].values
    s_vals = df[s].values

    # Sort by x
    idxs = np.argsort(x_vals)
    x_vals = x_vals[idxs]
    y_vals = y_vals[idxs]
    c_vals = c_vals[idxs]
    s_vals = s_vals[idxs]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        savereturn = True
    else:
        savereturn = False

    # if positive and negative values, add vertical x=0 line
    # if (x_vals.min() < 0) & (x_vals.max() > 0):
    ax.axvline(x=0, ls="--", color="lightgrey", lw=1.5)

    # determine scale factor based on maximum s_val
    scale = largest_dot / np.max(s_vals)
    ns = s_vals * scale * plt.rcParams["lines.markersize"]
    ax.grid(axis="x")
    scatter = ax.scatter(x=x_vals, y=y_vals, c=c_vals, s=ns, cmap=cmap)
    ax.set_axisbelow(True)
    ax.set_xlabel(x)

    font_size = "small"  # font size for colorbar and dot legend

    # Add legend
    handles, labels = scatter.legend_elements(
        prop="sizes",
        num=3,
        fmt="{x:.2f}",
        func=lambda s: s / plt.rcParams["lines.markersize"] / scale,
    )
    ax.legend(
        handles,
        labels,
        title=s,
        frameon=False,
        bbox_to_anchor=(1.0, 0.95),
        loc="upper left",
        labelspacing=1.0,
        fontsize=font_size,
        title_fontsize=font_size,
    )

    # Add colorbar
    clb = plt.colorbar(
        scatter,
        shrink=0.25,
        aspect=10,
        orientation="vertical",
        anchor=(1.0, 0.15),
        location="right",
    )
    clb.ax.set_title(c, loc="left", fontsize=font_size)
    clb.ax.tick_params(labelsize=font_size)

    ax.margins(x=0.25, y=0.1)

    if title is not None:
        ax.set_title(title)

    if savereturn:
        save_plot(fig, ax, save)

        if return_fig:
            return fig


def decoupler_dotplot_facet(
    df,
    group_col="group",
    x="Combined score",
    y="Term",
    c="FDR p-value",
    s="Overlap ratio",
    top_n=None,
    ncols=4,
    figsize_scale=1.5,
    save=None,
    dpi=200,
    **kwargs,
):
    """
    Plot results of `decoupler` enrichment analysis as dots, faceted by group

    Parameters
    ----------
    df : DataFrame
        results of enrichment analysis.
    group_col : str
        column from `df` to facet by
    x : str
        column name of `df` to use as continous value.
    y : str
        column name of `df` to use as labels.
    c : str
        column name of `df` to use for coloring.
    s : str
        column name of `df` to use for dot size.
    top_n : int
        number of top terms to plot per group. If `None` show all terms.
    ncols : int
        number of columns for faceting. If `None` use `len(df[group_col].unique())`
    figsize_scale : float
        scale size of `matplotlib` figure
    save : str
        path to file to save image to. If `None`, return axes objects
    dpi : float, optional (default=200)
        Resolution in dots per inch for saving figure. Ignored if `save` is `None`.
    **kwargs
        keyword arguments to pass to `decoupler_dotplot` function

    Returns
    -------
    fig : matplotlib.Figure
        Return figure object if `save==None`. Otherwise, write to `save`.
    """
    if ncols is None:
        ncols = len(df[group_col].unique())

    # generate gs object
    gs, fig = build_gridspec(
        panels=list(df[group_col].unique()),
        ncols=ncols,
        panelsize=(1.5 * ncols * figsize_scale, ncols * figsize_scale),
    )

    # add plots to axes
    for igroup, group in enumerate(df[group_col].unique()):
        tmp = df.loc[df[group_col] == group]
        if top_n is not None:
            tmp = tmp.nlargest(top_n, x, keep="all")
        _ax = plt.subplot(gs[igroup])
        decoupler_dotplot(
            df=tmp,
            x=x,
            y=y,
            c=c,
            s=s,
            title=group,
            ax=_ax,
            **kwargs,
        )

    gs.tight_layout(fig)
    if save is None:
        return fig
    else:
        print("Saving to {}".format(save))
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
