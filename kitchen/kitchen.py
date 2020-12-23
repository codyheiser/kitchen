# -*- coding: utf-8 -*-
"""
Manipulate .h5ad files and cook scRNA-seq data from command line

@author: C Heiser
"""
import argparse, os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy import sparse
from dropkick import recipe_dropkick

from .ingredients import (
    check_dir_exists,
    cellranger2,
    cellranger3,
    subset_adata,
    cc_score,
    dim_reduce,
    plot_embedding,
    plot_genes,
    plot_genes_cnmf,
    rank_genes_cnmf,
    cluster_pie,
)
from ._version import get_versions


def info(args):
    """Print information about .h5ad file to console"""
    print("Reading {}\n".format(args.file))
    adata = sc.read(args.file)
    print(adata, "\n")
    print(".X: {} with {}\n".format(type(adata.X), adata.X.dtype))
    print("obs_names: {}".format(adata.obs_names))
    print("var_names: {}".format(adata.var_names))


def to_h5ad(args):
    """Convert counts matrix from flat file (.txt, .csv) to .h5ad"""
    # get basename of file for writing outputs
    name = os.path.splitext(os.path.basename(args.file))[0]
    # check to make sure it's an .h5ad file
    if os.path.splitext(args.file)[1] == ".h5ad":
        raise ValueError("Input file already in .h5ad format")
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file), end="")
    a = sc.read(args.file)
    if args.verbose:
        # print information about counts, including names of cells and genes
        print(" - {} cells and {} genes".format(a.shape[0], a.shape[1]))
        print("obs_names: {}".format(a.obs_names))
        print("var_names: {}".format(a.var_names))
    # sparsify counts slot
    if args.verbose:
        print("sparsifying counts...")
    a.X = sparse.csr_matrix(a.X, dtype=int)
    # save file as .h5ad
    if args.verbose:
        print("Writing counts to {}/{}.h5ad".format(args.outdir, name))
    check_dir_exists(args.outdir)
    a.write("{}/{}.h5ad".format(args.outdir, name), compression="gzip")
    if args.rm_flat_file:
        # remove original, noncompressed flat file
        if args.verbose:
            print("Removing {}".format(args.file))
        os.remove(args.file)


def h5ad_to_csv(args):
    """Convert counts matrix from .h5ad to flat file (.txt, .csv)"""
    # get basename of file for writing outputs
    name = os.path.splitext(os.path.basename(args.file))[0]
    # check to make sure it's an .h5ad file
    if os.path.splitext(args.file)[1] != ".h5ad":
        raise ValueError("Input file must be in .h5ad format")
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file), end="")
    a = sc.read(args.file)
    if args.verbose:
        # print information about counts, including names of cells and genes
        print(" - {} cells and {} genes".format(a.shape[0], a.shape[1]))
    # swap to desired layer
    if args.layer is not None:
        if args.verbose:
            print("Using .layers['{}']".format(args.layer))
        a.X = a.layers[args.layer].copy()
    # check for/create output directory
    check_dir_exists(args.outdir)
    if args.separate_indices:
        if args.verbose:
            print("Writing counts to {}/{}_X.csv".format(args.outdir, name))
        if isinstance(a.X, sparse.csr.csr_matrix):
            df = pd.DataFrame.sparse.from_spmatrix(a.X)
            df.to_csv(
                "{}/{}_X.csv".format(args.outdir, name),
                sep=",",
                header=False,
                index=False,
            )
        elif isinstance(a.X, np.ndarray):
            np.savetxt("{}/{}_X.csv".format(args.outdir, name), a.X, delimiter=",")
        if args.verbose:
            print("Writing obs names to {}/{}_obs.csv".format(args.outdir, name))
        pd.DataFrame(a.obs_names).to_csv(
            "{}/{}_obs.csv".format(args.outdir, name), header=False, index=False
        )
        if args.verbose:
            print("Writing var names to {}/{}_var.csv".format(args.outdir, name))
        pd.DataFrame(a.var_names).to_csv(
            "{}/{}_var.csv".format(args.outdir, name), header=False, index=False
        )
    else:
        if isinstance(a.X, sparse.csr.csr_matrix):
            df = pd.DataFrame.sparse.from_spmatrix(
                a.X, index=a.obs_names, columns=a.var_names
            )
        else:
            df = pd.DataFrame(a.X, index=a.obs_names, columns=a.var_names)
        # save file as .csv
        if args.verbose:
            print("Writing counts to {}/{}.csv".format(args.outdir, name))
        df.to_csv(
            "{}/{}.csv".format(args.outdir, name), sep=",", header=True, index=True
        )
    if args.rm_h5ad_file:
        # remove original, h5ad file
        if args.verbose:
            print("Removing {}".format(args.file))
        os.remove(args.file)


def transpose(args):
    """Transpose anndata object, replacing obs with var, and overwrite .h5ad file"""
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file))
    a = sc.read(args.file)
    if args.verbose:
        print(a)
    # transpose file
    if args.verbose:
        print("transposing file and saving...")
    a = a.T
    # save file as .h5ad
    a.write(args.file, compression="gzip")


def to_sparse(args):
    """Convert .X slot of anndata object to scipy CSR format, overwrite .h5ad file"""
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file))
    a = sc.read(args.file)
    if args.verbose:
        print(a)
    if isinstance(a.X, sparse.csr.csr_matrix):
        print("{} already in sparse format".format(args.file))
        return
    # sparsify counts slot
    if args.verbose:
        print("sparsifying counts...")
    a.X = sparse.csr_matrix(a.X, dtype=int)
    # save file as .h5ad
    a.write(args.file, compression="gzip")


def to_dense(args):
    """Convert .X slot of anndata object to numpy.matrix format, overwrite .h5ad file"""
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file))
    a = sc.read(args.file)
    if args.verbose:
        print(a)
    if isinstance(a.X, np.matrix) or isinstance(a.X, np.ndarray):
        print("{} already in dense format".format(args.file))
        return
    # densify counts slot
    if args.verbose:
        print("densifying counts...")
    a.X = a.X.todense()
    a.X = a.X.astype(int)
    # save file as .h5ad
    a.write(args.file, compression="gzip")


def to_X(args):
    """Swap a matrix from .layers to .X slot of anndata object, overwrite .h5ad file"""
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file))
    a = sc.read(args.file)
    if args.verbose:
        print(a)
    # swap layers
    if args.verbose:
        print("Putting .layers['{}'] in .X and saving".format(args.layer))
    a.X = a.layers[args.layer].copy()
    # save file as .h5ad
    a.write(args.file, compression="gzip")


def rename_obs(args):
    """Rename .obs columns in anndata object, and overwrite .h5ad file"""
    if args.verbose:
        print("Reading {}".format(args.file))
    adata = sc.read(args.file)
    if args.verbose:
        print("Renaming columns {} to {}".format(args.old_names, args.new_names))
    adata.obs.rename(columns=dict(zip(args.old_names, args.new_names)), inplace=True)
    adata.write(args.file, compression="gzip")


def label_info(args):
    """Print value counts for .obs labels to console"""
    print("Reading {}\n".format(args.file))
    adata = sc.read(args.file)
    print(adata, "\n")
    for l in args.labels:
        print("{}\n{}\n".format(l, adata.obs[l].value_counts()))


def obs_to_categorical(args):
    """Make .obs label categorical dtype"""
    if args.verbose:
        print("Reading {}".format(args.file))
    adata = sc.read(args.file)
    for label in args.labels:
        if args.verbose:
            print("Converting .obs['{}'] to categorical".format(label))
        if args.to_bool:
            adata.obs[label] = adata.obs[label].astype(bool).astype("category")
        else:
            adata.obs[label] = adata.obs[label].astype("category")
    adata.write(args.file, compression="gzip")


def add_label(args):
    """
    Use .obs_names from filtered counts matrix to add binary label to a reference
    anndata object, 1 = present in filt, 0 = not present. Overwrite reference .h5ad file.
    """
    # read reference file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.ref_file))
    a = sc.read(args.ref_file)
    if args.verbose:
        print("\t", a)
    # read query file into anndata obj
    if args.verbose:
        print("\nReading {}".format(args.filt_file))
    b = sc.read(args.filt_file)
    if args.verbose:
        print("\t", b)
    # add .obs column to ref_file
    a.obs[args.label] = 0
    a.obs.loc[b.obs_names, args.label] = 1
    if args.verbose:
        print(
            "\nTransferring labels to {}:\n{}".format(
                args.ref_file, a.obs[args.label].value_counts()
            )
        )
    # save file as .h5ad
    if args.verbose:
        print("\nWriting counts to {}".format(args.ref_file))
    a.write(args.ref_file, compression="gzip")
    if args.rm_orig_file:
        # remove filtered file
        if args.verbose:
            print("\nRemoving {}".format(args.filt_file))
        os.remove(args.filt_file)


def knee_point(args):
    """Label cells using "knee point" method from CellRanger 2.1"""
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file), end="")
    a = sc.read(args.file)
    if args.verbose:
        print(" - {} cells and {} genes".format(a.shape[0], a.shape[1]))
    # add knee_point label to anndata
    cellranger2(
        a,
        expected=args.expected,
        upper_quant=args.upper_quant,
        lower_prop=args.lower_prop,
        verbose=args.verbose,
    )
    # save file as .h5ad
    print("Writing counts to {}".format(args.file))
    a.write(args.file, compression="gzip")


def emptydrops(args):
    """Label cells using "emptydrops" method from CellRanger 3.0"""
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file), end="")
    a = sc.read(args.file)
    if args.verbose:
        print(" - {} cells and {} genes".format(a.shape[0], a.shape[1]))
    # add EmptyDrops label to anndata
    cellranger3(
        a,
        init_counts=args.init_counts,
        min_umi_frac_of_median=args.min_umi_frac,
        min_umis_nonambient=args.min_umi,
        max_adj_pvalue=args.max_adj_pval,
    )
    # save file as .h5ad
    if args.verbose:
        print("Writing counts to {}".format(args.file))
    a.write(args.file, compression="gzip")


def subset(args):
    """Subset anndata object on binary .obs label(s), save to new .h5ad file"""
    if args.verbose:
        print("Reading {}".format(args.file), end="")
    a = sc.read(args.file)
    if args.verbose:
        print(" - {} cells and {} genes".format(a.shape[0], a.shape[1]))
    a = subset_adata(a, subset=args.subset, verbose=args.verbose)
    if args.verbose:
        print("Writing subsetted counts to {}".format(args.out))
    a.write(args.out, compression="gzip")


def concatenate(args):
    """Concatenate list of anndata objects in .h5ad format, keeping union of genes"""
    # read first file
    if args.verbose:
        print("Reading {}".format(args.files[0]))
    adata_0 = sc.read(args.files[0])
    # read the rest of the files into list
    adatas = []
    for f in args.files[1:]:
        # read file into anndata obj
        if args.verbose:
            print("Reading {}".format(f))
        adatas.append(sc.read(f))
    # concatenate all files
    if args.verbose:
        print("Concatenating files...")
    concat = adata_0.concatenate(
        adatas,
        join="outer",
        batch_categories=[os.path.splitext(os.path.basename(x))[0] for x in args.files],
        fill_value=0,
    )
    if args.verbose:
        print(
            "Final shape: {} cells and {} genes".format(
                concat.shape[0], concat.shape[1]
            )
        )
    # save file as .h5ad
    if args.verbose:
        print("Writing counts to {}".format(args.out))
    concat.write(args.out, compression="gzip")


def recipe(args):
    """Full automated processing of scRNA-seq data"""
    # get basename of file for writing outputs
    name = [os.path.splitext(os.path.basename(args.file))[0]]
    if args.subset is not None:
        name.append("_".join(args.subset))
    if args.layer is not None:
        name.append(args.layer)
    if args.use_rep is not None:
        name.append(args.use_rep)
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file), end="")
    a = sc.read(args.file)
    if args.verbose:
        print(" - {} cells and {} genes".format(a.shape[0], a.shape[1]))
    # subset anndata on .obs column if desired
    if args.subset is not None:
        a = subset_adata(a, subset=args.subset)
    if args.process:
        # switch to proper layer
        if args.layer is not None:
            if args.verbose:
                print("Using layer {} to reduce dimensions".format(args.layer))
            a.X = a.layers[args.layer].copy()
        # preprocess with dropkick recipe
        a = recipe_dropkick(
            a,
            X_final="arcsinh_norm",
            verbose=args.verbose,
            filter=True,
            min_genes=args.min_genes,
        )
        # reduce dimensions
        dim_reduce(
            a,
            use_rep=args.use_rep,
            clust_resolution=args.resolution,
            paga=args.paga,
            verbose=args.verbose,
            seed=args.seed,
        )
    # run cell cycle inference
    if args.cell_cycle:
        cc_score(a, verbose=args.verbose)
        args.colors = ["phase"] + args.colors
    # make sure output dir exists before saving plots
    check_dir_exists(args.outdir)
    # if there's DE to do, plot genes
    if args.diff_expr is not None:
        wd = os.getcwd()  # save current working directory for later
        os.chdir(args.outdir)  # set output directory for scanpy figures
        plot_genes(
            a,
            plot_type=args.diff_expr,
            groupby="leiden",
            n_genes=5,
            cmap=args.cmap,
            save_to="_{}.png".format("_".join(name)),
            verbose=args.verbose,
        )
        # if there's cnmf results, plot those on a heatmap/matrix/dotplot too
        if "cnmf_spectra" in a.varm:
            plot_genes_cnmf(
                a,
                plot_type=args.diff_expr,
                groupby="leiden",
                attr="varm",
                keys="cnmf_spectra",
                indices=None,
                n_genes=5,
                cmap=args.cmap,
                save_to="_cnmf_{}.png".format("_".join(name)),
            )
        os.chdir(wd)  # go back to previous working directory after saving scanpy plots
    # if there's a cnmf flag, try to plot loadings
    if args.cnmf:
        # check for cnmf results in anndata object
        if "cnmf_spectra" in a.varm:
            _ = rank_genes_cnmf(a, show=False)
            if args.verbose:
                print(
                    "Saving cNMF loadings to {}/{}_cnmfspectra.png".format(
                        args.outdir, "_".join(name)
                    )
                )
            plt.savefig("{}/{}_cnmfspectra.png".format(args.outdir, "_".join(name)))
            if args.verbose:
                print(
                    "Saving embeddings to {}/{}_embedding.png".format(
                        args.outdir, "_".join(name)
                    )
                )
            # save embedding plot with cNMF loadings
            if args.colors is None:
                args.colors = []
            plot_embedding(
                a,
                colors=args.colors
                + a.obs.columns[a.obs.columns.str.startswith("usage_")].tolist(),
                show_clustering=True,
                n_cnmf_markers=args.n_cnmf_markers,
                cmap=args.cmap,
                save_to="{}/{}_embedding.png".format(args.outdir, "_".join(name)),
                verbose=args.verbose,
            )
        else:
            print(
                "cNMF results not detected in {}. Skipping cNMF overlay for embedding.".format(
                    args.file
                )
            )
            # save embedding plot without cNMF loadings
            if args.verbose:
                print(
                    "Saving embeddings to {}/{}_embedding.png".format(
                        args.outdir, "_".join(name)
                    )
                )
            plot_embedding(
                a,
                colors=args.colors,
                show_clustering=True,
                cmap=args.cmap,
                save_to="{}/{}_embedding.png".format(args.outdir, "_".join(name)),
                verbose=args.verbose,
            )
    else:
        # save embedding plot
        if args.verbose:
            print(
                "Saving embeddings to {}/{}_embedding.png".format(
                    args.outdir, "_".join(name)
                )
            )
        plot_embedding(
            a,
            colors=args.colors,
            show_clustering=True,
            cmap=args.cmap,
            save_to="{}/{}_embedding.png".format(args.outdir, "_".join(name)),
            verbose=args.verbose,
        )
    # save file as .h5ad
    if args.save_adata:
        if args.verbose:
            print(
                "Saving AnnData object to to {}/{}_processed.h5ad".format(
                    args.outdir, "_".join(name)
                )
            )
        a.write(
            "{}/{}_processed.h5ad".format(args.outdir, "_".join(name)),
            compression="gzip",
        )


def de(args):
    """Perform differential expression analysis on a processed .h5ad file and plot results"""
    # get basename of file for writing outputs
    name = [os.path.splitext(os.path.basename(args.file))[0]]
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file), end="")
    a = sc.read(args.file)
    if args.verbose:
        print(" - {} cells and {} genes".format(a.shape[0], a.shape[1]))
    # perform DE analysis and plot genes
    os.chdir(args.outdir)  # set output directory for scanpy figures
    plot_genes(
        a,
        plot_type=args.plot_type,
        groupby=args.groupby,
        n_genes=args.n_genes,
        ambient=args.ambient,
        dendrogram=args.dendrogram,
        cmap=args.cmap,
        save_to="_{}.png".format("_".join(name)),
        verbose=args.verbose,
    )


def cnmf_markers(args):
    """Plot heatmap/matrix/dotplot of cNMF loadings for desired groups"""
    # get basename of file for writing outputs
    name = [os.path.splitext(os.path.basename(args.file))[0]]
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file), end="")
    a = sc.read(args.file)
    if args.verbose:
        print(" - {} cells and {} genes".format(a.shape[0], a.shape[1]))
    # plot cNMF marker genes
    os.chdir(args.outdir)  # set output directory for scanpy figures
    plot_genes_cnmf(
        a,
        plot_type=args.plot_type,
        groupby=args.groupby,
        n_genes=args.n_genes,
        dendrogram=args.dendrogram,
        cmap=args.cmap,
        save_to="_cnmf_{}.png".format("_".join(name)),
    )


def pie(args):
    """plot populational pie charts for desired groups"""
    # get basename of file for writing outputs
    name = [os.path.splitext(os.path.basename(args.file))[0]]
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file), end="")
    a = sc.read(args.file)
    if args.verbose:
        print(" - {} cells and {} genes".format(a.shape[0], a.shape[1]))
    # generate cluster_pie plot
    os.chdir(args.outdir)  # set output directory for scanpy figures
    _ = cluster_pie(a, pie_by=args.pieby, groupby=args.groupby)
    if args.verbose:
        print(
            "Saving cluster pie charts to {}/{}_pie.png".format(
                args.outdir, "_".join(name)
            )
        )
    plt.savefig("{}/{}_pie.png".format(args.outdir, "_".join(name)))


def main():
    parser = argparse.ArgumentParser(prog="kitchen")
    parser.add_argument(
        "-V", "--version", action="version", version=get_versions()["version"],
    )
    subparsers = parser.add_subparsers()

    info_parser = subparsers.add_parser(
        "info", help="Show information about .h5ad file",
    )
    info_parser.add_argument(
        "file", type=str, help="Counts matrix as .h5ad file",
    )
    info_parser.set_defaults(func=info)

    to_h5ad_parser = subparsers.add_parser(
        "to_h5ad", help="Convert counts matrix to .h5ad format",
    )
    to_h5ad_parser.add_argument(
        "file", type=str, help="Counts matrix as comma or tab delimited text file",
    )
    to_h5ad_parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        nargs="?",
        default=".",
        help="Output directory for writing h5ad. Default './'",
    )
    to_h5ad_parser.add_argument(
        "-rm",
        "--rm-flat-file",
        help="Remove original flat file. Default False",
        action="store_true",
    )
    to_h5ad_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    to_h5ad_parser.set_defaults(func=to_h5ad)

    to_csv_parser = subparsers.add_parser(
        "to_csv", help="Save .h5ad counts to .csv file(s)",
    )
    to_csv_parser.add_argument(
        "file", type=str, help="Counts matrix as .h5ad file",
    )
    to_csv_parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        nargs="?",
        default=".",
        help="Output directory for writing csv file(s). Default './'",
    )
    to_csv_parser.add_argument(
        "-l",
        "--layer",
        type=str,
        default=None,
        help="Key from .layers to save. Default '.X'.",
    )
    to_csv_parser.add_argument(
        "-s",
        "--separate-indices",
        help="Save indices (.obs and .var names) to separate files",
        action="store_true",
    )
    to_csv_parser.add_argument(
        "-rm",
        "--rm-h5ad-file",
        help="Remove original h5ad file. Default False",
        action="store_true",
    )
    to_csv_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    to_csv_parser.set_defaults(func=h5ad_to_csv)

    transpose_parser = subparsers.add_parser(
        "transpose", help="Transpose counts matrix and save as .h5ad",
    )
    transpose_parser.add_argument(
        "file",
        type=str,
        help="Input counts matrix as .h5ad or tab delimited text file",
    )
    transpose_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    transpose_parser.set_defaults(func=transpose)

    to_sparse_parser = subparsers.add_parser(
        "to_sparse",
        help="Convert .X slot of anndata object to scipy CSR format, overwrite .h5ad file",
    )
    to_sparse_parser.add_argument(
        "file",
        type=str,
        help="Input counts matrix as .h5ad or tab delimited text file",
    )
    to_sparse_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    to_sparse_parser.set_defaults(func=to_sparse)

    to_dense_parser = subparsers.add_parser(
        "to_dense",
        help="Convert .X slot of anndata object to numpy.matrix format, overwrite .h5ad file",
    )
    to_dense_parser.add_argument(
        "file",
        type=str,
        help="Input counts matrix as .h5ad or tab delimited text file",
    )
    to_dense_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    to_dense_parser.set_defaults(func=to_dense)

    to_X_parser = subparsers.add_parser(
        "to_X", help="Swap a matrix from .layers to .X slot of anndata object",
    )
    to_X_parser.add_argument(
        "file", type=str, help="Counts matrix as .h5ad file",
    )
    to_X_parser.add_argument(
        "-l",
        "--layer",
        type=str,
        required=True,
        help="Key from .layers to replace .X with",
    )
    to_X_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    to_X_parser.set_defaults(func=to_X)

    rename_obs_parser = subparsers.add_parser(
        "rename_obs", help="Rename .obs columns in .h5ad file",
    )
    rename_obs_parser.add_argument(
        "file", type=str, help="Counts matrix as .h5ad file",
    )
    rename_obs_parser.add_argument(
        "-o",
        "--old-names",
        type=str,
        nargs="+",
        required=True,
        help="List of existing .obs column names to change",
    )
    rename_obs_parser.add_argument(
        "-n",
        "--new-names",
        type=str,
        nargs="+",
        required=True,
        help="List of new .obs column names corresponding to args.old_names",
    )
    rename_obs_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    rename_obs_parser.set_defaults(func=rename_obs)

    label_info_parser = subparsers.add_parser(
        "label_info", help="Print value counts for .obs labels to console",
    )
    label_info_parser.add_argument(
        "file", type=str, help="Counts matrix as .h5ad file",
    )
    label_info_parser.add_argument(
        "-l",
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="List of .obs column names to print value counts for",
    )
    label_info_parser.set_defaults(func=label_info)

    to_categorical_parser = subparsers.add_parser(
        "to_categorical", help="Make .obs label categorical dtype",
    )
    to_categorical_parser.add_argument(
        "file", type=str, help="Counts matrix as .h5ad file",
    )
    to_categorical_parser.add_argument(
        "-l",
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="List of .obs column names to convert to categorical dtype",
    )
    to_categorical_parser.add_argument(
        "-b",
        "--to-bool",
        required=False,
        help="Convert to boolean first (for columns with values of {0,1})",
        action="store_true",
    )
    to_categorical_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    to_categorical_parser.set_defaults(func=obs_to_categorical)

    add_label_parser = subparsers.add_parser(
        "add_label",
        help="Add label to .obs of .h5ad reference file using cells in a filtered file",
    )
    add_label_parser.add_argument(
        "ref_file",
        type=str,
        help="Reference counts matrix to add label to, as .h5ad file",
    )
    add_label_parser.add_argument(
        "filt_file",
        type=str,
        help="Filtered counts matrix providing positive labels, as .h5ad file",
    )
    add_label_parser.add_argument(
        "-l",
        "--label",
        type=str,
        required=True,
        help=".obs column name to place final labels in",
    )
    add_label_parser.add_argument(
        "-rm",
        "--rm-orig-file",
        help="Remove filtered file. Default False",
        action="store_true",
    )
    add_label_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    add_label_parser.set_defaults(func=add_label)

    cellranger2_parser = subparsers.add_parser(
        "cellranger2", help="Label cells using 'knee point' method from CellRanger 2.1",
    )
    cellranger2_parser.add_argument(
        "file", type=str, help="Counts matrix as .h5ad file",
    )
    cellranger2_parser.add_argument(
        "-e",
        "--expected",
        type=int,
        nargs="?",
        default=3000,
        help="Number of expected cells in the dataset; default 3000",
    )
    cellranger2_parser.add_argument(
        "-u",
        "--upper-quant",
        type=float,
        default=0.99,
        help="Upper quantile of expected cells for knee point; default 0.99",
    )
    cellranger2_parser.add_argument(
        "-l",
        "--lower-prop",
        type=float,
        default=0.1,
        help="Lower proportion of cells to set threshold at; default 0.1",
    )
    cellranger2_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    cellranger2_parser.set_defaults(func=knee_point)

    cellranger3_parser = subparsers.add_parser(
        "cellranger3", help="Label cells using 'emptydrops' method from CellRanger 3.0",
    )
    cellranger3_parser.add_argument(
        "file", type=str, help="Counts matrix as .h5ad file",
    )
    cellranger3_parser.add_argument(
        "--init-counts",
        type=int,
        nargs="?",
        default=15000,
        help="Initial total counts threshold for calling cells; default 15000",
    )
    cellranger3_parser.add_argument(
        "--min-umi-frac",
        type=float,
        default=0.01,
        help="Minimum total counts for testing barcodes as fraction of median counts for initially labeled cells; default 0.01",
    )
    cellranger3_parser.add_argument(
        "--min-umi",
        type=int,
        default=500,
        help="Minimum total counts for testing barcodes; default 500",
    )
    cellranger3_parser.add_argument(
        "--max-adj-pval",
        type=float,
        default=0.01,
        help="Maximum p-value for cell calling after B-H correction; default 0.01",
    )
    cellranger3_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    cellranger3_parser.set_defaults(func=emptydrops)

    subset_parser = subparsers.add_parser(
        "subset", help="Subset AnnData object on one or more .obs columns",
    )
    subset_parser.add_argument(
        "file", type=str, help="Counts matrix as .h5ad file",
    )
    subset_parser.add_argument(
        "-s",
        "--subset",
        default=None,
        nargs="*",
        help=".obs column(s) to subset cells on",
    )
    subset_parser.add_argument(
        "-o",
        "--out",
        type=str,
        nargs="?",
        default="./subset.h5ad",
        help="Path to output .h5ad file. Default './subset.h5ad'",
    )
    subset_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    subset_parser.set_defaults(func=subset)

    concatenate_parser = subparsers.add_parser(
        "concatenate", help="Combine multiple .h5ad files",
    )
    concatenate_parser.add_argument(
        "files", type=str, nargs="*", help="List of .h5ad files to concatenate"
    )
    concatenate_parser.add_argument(
        "-o",
        "--out",
        type=str,
        nargs="?",
        default="./concat.h5ad",
        help="Path to output .h5ad file. Default './concat.h5ad'",
    )
    concatenate_parser.add_argument(
        "-q",
        "--quietly",
        required=False,
        help="Run without printing processing updates to console",
        action="store_true",
    )
    concatenate_parser.set_defaults(func=concatenate)

    recipe_parser = subparsers.add_parser(
        "recipe", help="Full automated processing of scRNA-seq data from command line",
    )
    recipe_parser.add_argument(
        "file",
        type=str,
        help="Counts file as .h5ad or flat (.csv, .txt) in cells x genes format",
    )
    recipe_parser.add_argument(
        "-p",
        "--process",
        help="Process AnnData (PCA, PAGA, UMAP). Default False",
        action="store_true",
    )
    recipe_parser.add_argument(
        "--min-genes",
        required=False,
        type=int,
        default=1,
        help="Minimum number of genes detected to keep cell. Default 1.",
    )
    recipe_parser.add_argument(
        "-s",
        "--subset",
        nargs="*",
        default=None,
        help=".obs column(s) to subset cells on before embedding",
    )
    recipe_parser.add_argument(
        "-l",
        "--layer",
        type=str,
        default=None,
        help="Key from .layers to use for embedding. Default '.X'.",
    )
    recipe_parser.add_argument(
        "-ur",
        "--use-rep",
        type=str,
        default=None,
        help="Key from .obsm to use for neighbors graph and embedding. Default run PCA.",
    )
    recipe_parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution between 0.0 and 1.0 for Leiden clustering. Default 1.0.",
    )
    recipe_parser.add_argument(
        "-cc",
        "--cell-cycle",
        help="Calculate cell cycle scores. Default False",
        action="store_true",
    )
    recipe_parser.add_argument(
        "--paga",
        help="Run PAGA to seed UMAP embedding. Default False",
        action="store_true",
    )
    recipe_parser.add_argument(
        "-de",
        "--diff_expr",
        type=str,
        nargs="*",
        default=None,
        help="Type(s) of DE gene expression plots ['heatmap', 'dotplot', 'matrixplot']",
    )
    recipe_parser.add_argument(
        "-c",
        "--colors",
        type=str,
        nargs="*",
        default=[],
        help="Colors to plot on embedding. Can be .obs columns or gene names.",
    )
    recipe_parser.add_argument(
        "-cm",
        "--cmap",
        required=False,
        type=str,
        default="Reds",
        help="Color map to use in UMAP overlays and genes plot",
    )
    recipe_parser.add_argument(
        "--cnmf",
        help="Plot cNMF usages on embedding. Default False.",
        action="store_true",
    )
    recipe_parser.add_argument(
        "--n-cnmf-markers",
        required=False,
        type=int,
        default=7,
        help="Number of top loaded genes to print on cNMF embeddings. Default 7.",
    )
    recipe_parser.add_argument(
        "--seed",
        type=int,
        help="Random state for generating embeddings. Default 18",
        default=18,
    )
    recipe_parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        nargs="?",
        default=".",
        help="Output directory for saving plots. Default './'",
    )
    recipe_parser.add_argument(
        "-sa",
        "--save-adata",
        help="Save updated AnnData to .h5ad file. Default False",
        action="store_true",
    )
    recipe_parser.add_argument(
        "-q", "--quietly", help="Don't print updates to console", action="store_true",
    )
    recipe_parser.set_defaults(func=recipe)

    de_parser = subparsers.add_parser(
        "de",
        help="Perform differential expression analysis on a processed .h5ad file and plot results",
    )
    de_parser.add_argument(
        "file",
        type=str,
        help="Counts file as .h5ad or flat (.csv, .txt) in cells x genes format",
    )
    de_parser.add_argument(
        "-p",
        "--plot-type",
        type=str,
        nargs="*",
        default=None,
        help="Type(s) of DE gene expression plot ['heatmap', 'dotplot', 'matrixplot']",
    )
    de_parser.add_argument(
        "-g",
        "--groupby",
        required=False,
        type=str,
        default="leiden",
        help=".obs variable to group cells by for DE analysis",
    )
    de_parser.add_argument(
        "-n",
        "--n-genes",
        type=int,
        default=5,
        help="Number of genes to plot per group. Default 5.",
    )
    de_parser.add_argument(
        "-cm",
        "--cmap",
        required=False,
        type=str,
        default="Reds",
        help="Color map to use in genes plot",
    )
    de_parser.add_argument(
        "-a", "--ambient", help="Include ambient genes", action="store_true",
    )
    de_parser.add_argument(
        "-d",
        "--dendrogram",
        help="Generate dendrogram of group similarities",
        action="store_true",
    )
    de_parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Output directory for saving plots. Default './'",
        nargs="?",
        default=".",
    )
    de_parser.add_argument(
        "-q", "--quietly", help="Don't print updates to console", action="store_true",
    )
    de_parser.set_defaults(func=de)

    pie_parser = subparsers.add_parser(
        "pie", help="Plot populational pie charts for desired groups",
    )
    pie_parser.add_argument(
        "file",
        type=str,
        help="Counts file as .h5ad or flat (.csv, .txt) in cells x genes format",
    )
    pie_parser.add_argument(
        "-g",
        "--groupby",
        required=False,
        type=str,
        default="leiden",
        help=".obs variable to group cells by for plotting pie charts",
    )
    pie_parser.add_argument(
        "-p",
        "--pieby",
        required=False,
        type=str,
        default="batch",
        help=".obs variable to group cells by for pie chart within each groupby category",
    )
    pie_parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        nargs="?",
        default=".",
        help="Output directory for saving plots. Default './'",
    )
    pie_parser.add_argument(
        "-q", "--quietly", help="Don't print updates to console", action="store_true",
    )
    pie_parser.set_defaults(func=pie)

    cnmf_markers_parser = subparsers.add_parser(
        "cnmf_markers",
        help="Plot heatmap/matrix/dotplot of cNMF loadings for desired groups",
    )
    cnmf_markers_parser.add_argument(
        "file",
        type=str,
        help="Counts file as .h5ad or flat (.csv, .txt) in cells x genes format",
    )
    cnmf_markers_parser.add_argument(
        "-p",
        "--plot-type",
        type=str,
        nargs="*",
        default=None,
        help="Type(s) of gene expression plot ['heatmap', 'dotplot', 'matrixplot']",
    )
    cnmf_markers_parser.add_argument(
        "-g",
        "--groupby",
        required=False,
        type=str,
        default="leiden",
        help=".obs variable to group cells by for plotting cNMF genes",
    )
    cnmf_markers_parser.add_argument(
        "-n",
        "--n-genes",
        type=int,
        default=5,
        help="Number of genes to plot per group. Default 5.",
    )
    cnmf_markers_parser.add_argument(
        "-cm",
        "--cmap",
        required=False,
        type=str,
        default="Reds",
        help="Color map to use in genes plot",
    )
    cnmf_markers_parser.add_argument(
        "-d",
        "--dendrogram",
        help="Generate dendrogram of group similarities",
        action="store_true",
    )
    cnmf_markers_parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        nargs="?",
        default=".",
        help="Output directory for saving plots. Default './'",
    )
    cnmf_markers_parser.add_argument(
        "-q", "--quietly", help="Don't print updates to console", action="store_true",
    )
    cnmf_markers_parser.set_defaults(func=cnmf_markers)

    args = parser.parse_args()

    # if --quietly specified, reverse verbosity
    if hasattr(args, "quietly"):
        if args.quietly:
            args.verbose = False
            del args.quietly
        else:
            args.verbose = True
            del args.quietly

    args.func(args)
