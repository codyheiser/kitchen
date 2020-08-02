# -*- coding: utf-8 -*-
"""
Manipulate .h5ad files and cook scRNA-seq data from command line

@author: C Heiser
"""
import argparse, os
import scanpy as sc
import matplotlib.pyplot as plt
from dropkick import recipe_dropkick

from .ingredients import (
    check_dir_exists,
    cellranger2,
    cellranger3,
    subset_adata,
    cc_score,
    dim_reduce,
    plot_embedding,
    rank_genes_cnmf,
)
from ._version import get_versions


def info(args):
    """Print information about .h5ad file to console"""
    print("Reading {}\n".format(args.file))
    adata = sc.read(args.file)
    print(adata, "\n")
    print("obs_names: {}".format(adata.obs_names))
    print("var_names: {}".format(adata.var_names))


def to_h5ad(args):
    """Convert counts matrix from flat file (.txt, .csv) to .h5ad"""
    # get basename of file for writing outputs
    name = os.path.splitext(os.path.basename(args.file))[0]
    # read file into anndata obj
    if args.verbose:
        print("Reading {}".format(args.file), end="")
    a = sc.read(args.file)
    if args.verbose:
        # print information about counts, including names of cells and genes
        print(" - {} cells and {} genes".format(a.shape[0], a.shape[1]))
        print("obs_names: {}".format(a.obs_names))
        print("var_names: {}".format(a.var_names))
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


def rename_obs(args):
    """Rename .obs columns in anndata object, and overwrite .h5ad file"""
    if args.verbose:
        print("Reading {}".format(args.file))
    adata = sc.read(args.file)
    if args.verbose:
        print("Renaming columns {} to {}".format(args.old_names, args.new_names))
    adata.obs.rename(columns=dict(zip(args.old_names, args.new_names)), inplace=True)
    adata.write(args.file, compression="gzip")


def add_label(args):
    """
    Use .obs_names from filtered counts matrix to add binary label to a reference
    anndata object, 1 = present in filt, 0 = not present. Overwrite reference .h5ad file.
    """
    # get basename of file for writing outputs
    name = os.path.splitext(os.path.basename(args.ref_file))[0]
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
    a.obs[args.obs_name] = 0
    a.obs.loc[b.obs_names, args.obs_name] = 1
    print(
        "\nTransferring labels to {}:\n{}".format(
            args.ref_file, a.obs[args.obs_name].value_counts()
        )
    )
    # save file as .h5ad
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
        # print information about counts, including names of cells and genes
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
        # print information about counts, including names of cells and genes
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
    print("Writing counts to {}".format(args.file))
    a.write(args.file, compression="gzip")


def subset(args):
    """Subset anndata object on binary .obs label(s), save to new .h5ad file"""
    a = subset_adata(args.file, subset=args.subset, verbose=args.verbose)
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
            rint("Reading {}".format(f))
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
            verbose=args.verbose,
            seed=args.seed,
        )
    # run cell cycle inference
    if args.cell_cycle:
        cc_score(a, verbose=args.verbose)
    # make sure output dir exists before saving plots
    check_dir_exists(args.outdir)
    # if there's cnmf results, plot loadings
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
        help="Output directory for writing h5ad. Default './'",
        nargs="?",
        default=".",
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
        help="Number of expected cells in the dataset; default 3000",
        nargs="?",
        default=3000,
    )
    cellranger2_parser.add_argument(
        "-u",
        "--upper-quant",
        type=float,
        help="Upper quantile of expected cells for knee point; default 0.99",
        default=0.99,
    )
    cellranger2_parser.add_argument(
        "-l",
        "--lower-prop",
        type=float,
        help="Lower proportion of cells to set threshold at; default 0.1",
        default=0.1,
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
        help="Initial total counts threshold for calling cells; default 15000",
        nargs="?",
        default=15000,
    )
    cellranger3_parser.add_argument(
        "--min-umi-frac",
        type=float,
        help="Minimum total counts for testing barcodes as fraction of median counts for initially labeled cells; default 0.01",
        default=0.01,
    )
    cellranger3_parser.add_argument(
        "--min-umi",
        type=int,
        help="Minimum total counts for testing barcodes; default 500",
        default=500,
    )
    cellranger3_parser.add_argument(
        "--max-adj-pval",
        type=float,
        help="Maximum p-value for cell calling after B-H correction; default 0.01",
        default=0.01,
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
        help="Path to output .h5ad file. Default './subset.h5ad'",
        nargs="?",
        default="./subset.h5ad",
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
        help="Path to output .h5ad file. Default './concat.h5ad'",
        nargs="?",
        default="./concat.h5ad",
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
        "--min-genes",
        required=False,
        type=int,
        help="Minimum number of genes detected to keep cell. Default 1.",
        default=1,
    )
    recipe_parser.add_argument(
        "-s",
        "--subset",
        default=None,
        nargs="*",
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
        help="Key from .obsm to use for neighbors graph and embedding. Default 'PCA'.",
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
        "-c",
        "--colors",
        type=str,
        help="Colors to plot on embedding. Can be .obs columns or gene names.",
        nargs="*",
        default=None,
    )
    recipe_parser.add_argument(
        "-p",
        "--process",
        help="Process AnnData (PCA, PAGA, UMAP). Default False",
        action="store_true",
    )
    recipe_parser.add_argument(
        "--seed", type=int, help="Random state for generating embeddings.", default=18,
    )
    recipe_parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Output directory for saving plots. Default './'",
        nargs="?",
        default=".",
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
