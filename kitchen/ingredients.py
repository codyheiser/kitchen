# -*- coding: utf-8 -*-
"""
Resources and utility functions
"""
import os, errno
import pandas as pd
import decoupler as dc
import liana as ln

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


def counts_to_RPKM(counts_mat, gene_lengths, mapped_reads=None):
    """
    Convert `counts_mat` to RPKM (reads per kilobase per million)

    Parameters
    ----------
    counts_mat : np.array
        Matrix of counts in samples x genes format (genes as columns)
    gene_lengths : np.array
        1D array of length `counts_mat.shape[1]` containing lengths for each gene in
        base pairs
    mapped_reads : np.array, optional (default=`None`)
        1D array of length `counts_mat.shape[0]` containing total mapped reads for each
        sample. If `None`, calculate sums manually from columns of `counts_mat`.

    Returns
    -------
    RPKM_mat : np.array
        Matrix in same shape as `counts_mat` containing RPKM values
    """
    RPK_mat = counts_mat / gene_lengths * 1e3  # reads per kilobase
    # reads per kilobase per million
    if mapped_reads is None:
        RPKM_mat = (RPK_mat.T / counts_mat.sum(axis=1) * 1e6).T
    else:
        RPKM_mat = (RPK_mat.T / mapped_reads * 1e6).T
    return RPKM_mat


def RPKM_to_TPM(RPKM_mat):
    """
    Convert `RPKM_mat` (reads per KB per million) to TPM (transcripts per million)

    Parameters
    ----------
    RPKM_mat : np.array
        Matrix of RPKM values in samples x genes format (genes as columns)

    Returns
    -------
    TPM_mat : np.array
        Matrix in same shape as `RPKM_mat` containing TPM values
    """
    TPM_mat = (RPKM_mat.T / RPKM_mat.sum(axis=1) * 1e6).T
    return TPM_mat


def counts_to_TPM(counts_mat, gene_lengths, mapped_reads=None):
    """
    Convert `counts_mat` to TPM (transcripts per million)

    Parameters
    ----------
    counts_mat : np.array
        Matrix of counts in samples x genes format (genes as columns)
    gene_lengths : np.array
        1D array of length `counts_mat.shape[1]` containing lengths for each gene in
        base pairs
    mapped_reads : np.array (default=`None`)
        1D array of length `counts_mat.shape[0]` containing total mapped reads for each
        sample. If `None`, calculate sums manually from columns of `counts_mat`.

    Returns
    -------
    TPM_mat : np.array
        Matrix in same shape as `counts_mat` containing TPM values
    """
    RPKM_mat = counts_to_RPKM(counts_mat, gene_lengths, mapped_reads)
    TPM_mat = RPKM_to_TPM(RPKM_mat)
    return TPM_mat


def human_to_mouse_simple(symbol):
    """Convert human to mouse symbols by simple case-conversion"""
    return "".join([symbol[0]] + [s.lower() for s in symbol[1:]])


def signatures_to_long_form(sig_short, sig_col="signature", gene_col="gene"):
    """
    Convert gene signatures dict or dataframe from short to long form

    Parameters
    ----------
    sig_short : Union[dict,pd.DataFrame]
        Gene signatures in dict or short form dataframe, where columns are assumed to
        contain separate signatures, with first row of column headers as signature
        names.
    sig_col : str, Optional (default='signature')
        Column in `sig_long` to contain signature names.
    gene_col : str, Optional (default='gene')
        Column in `sig_long` to contain gene names.

    Returns
    -------
    sig_long : pd.DataFrame
        Gene signatures in long form, with signature names in `sig_col` and gene names
        in `gene_col`.
    """
    if isinstance(sig_short, dict):
        print("Converting signature dict to pd.DataFrame")
        sig_short = pd.DataFrame(
            dict([(k, pd.Series(v)) for k, v in sig_short.items()])
        )
    # melt dataframe into long form
    sig_long = pd.melt(sig_short.fillna(0), value_name=gene_col, var_name=sig_col)
    sig_long = sig_long.loc[sig_long[gene_col].astype(str) != "0"]
    return sig_long


def signatures_to_short_form(sig_long, sig_col="signature", gene_col="gene"):
    """
    Convert gene signatures dict or dataframe from long to short form

    Parameters
    ----------
    sig_long : Union[dict,pd.DataFrame]
        Gene signatures in dict or long form dataframe, where `sig_col` and `gene_col`
        headers describe signature names and constituent genes, respectively.
    sig_col : str, Optional (default='signature')
        Column in `sig_long` containing signature names.
    gene_col : str, Optional (default='gene')
        Column in `sig_long` containing gene names.

    Returns
    -------
    sig_short : pd.DataFrame
        Gene signatures in short form, where columns are assumed to
        contain separate signatures, with first row of column headers as signature
        names.
    """
    if isinstance(sig_long, pd.DataFrame):
        print("Converting pd.DataFrame to signature dict")
        sig_long = (
            sig_long.groupby([sig_col])[gene_col].agg(lambda grp: list(grp)).to_dict()
        )
    # cast to short form from dictionary
    sig_short = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in sig_long.items()]))
    return sig_short


def ingest_gene_signatures(
    sig_files,
    form="short",
    sig_col="signature",
    gene_col="gene",
):
    """
    Read in gene signatures from one or more flat files

    Parameters
    ----------
    sig_files : Union[list, str]
        Path to single signature file or list of paths to signature files
    form : Literal('short','long'), Optional (default='short')
        Format of the data in `sig_files`. If 'short', columns of `sig_files` are
        assumed to contain separate signatures, with first row of column headers as
        signature names. If 'long', expect `sig_col` and `gene_col` headers to describe
        signature names and constituent genes in long-form, respectively.
    sig_col : str, Optional (default='signature')
        Column in `sig_files` containing signature names. Ignored if `form`=='short'.
    gene_col : str, Optional (default='gene')
        Column in `sig_files` containing gene names. Ignored if `form`=='short'.

    Returns
    -------
    genes : dict
        Dictionary of gene signatures with signature names as keys and lists of genes
        as values.
    """
    assert form in ("short", "long"), "form must be one of ('short','long')"
    # initialize dict
    genes = {}
    # coerce to list
    if isinstance(sig_files, str):
        sig_files = [sig_files]
    print("Ingesting gene signatures from {} flat files:".format(len(sig_files)))
    # loop through list of signature files
    for sig_file in sig_files:
        # determine file extension for delimiter
        filename, file_extension = os.path.splitext(sig_file)
        if file_extension in [".txt", ".tsv"]:
            sep = "\t"
        elif file_extension == ".csv":
            sep = ","
        else:
            raise ValueError("sig_files extensions must be one of (.txt, .tsv, .csv)")
        # read file
        g = pd.read_csv(sig_file, sep=sep)
        print("\tReading from {}".format(filename))
        # ingest short form
        if form == "short":
            g = g.fillna(0)  # ignore NaN
        # ingest long form
        elif form == "long":
            g = g.groupby([sig_col])[gene_col].agg(lambda grp: list(grp)).to_dict()
        new_sigs = 0
        for key in g.keys():
            # overwrite duplicates with warning
            if key in genes:
                print("\t\tOverwriting {} with signature from {}".format(key, filename))
            genes[key] = [x for x in g[key] if x != 0]
            new_sigs += 1

        print("\t\t{} gene signatures added".format(new_sigs))

    return genes


def signature_dict_from_rank_genes_groups(
    adata,
    uns_key="rank_genes_groups",
    groups=None,
    n_genes=5,
    ambient=False,
):
    """
    Extract DEGs from AnnData into signature dictionary

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing DEG results in `.uns`
    uns_key : str, optional (default='rank_genes_groups')
        Key from `adata.uns` containing DEG results. Should reflect categories in
        `adata.obs[groupby]`.
    groups : list of str, optional (default=`None`)
        List of groups within `adata.uns[uns_key]` to extract DEGs for. If `None`,
        retrieve all groups.
    n_genes : int, optional (default=5)
        Number of top genes per group to show
    ambient : bool, optional (default=False)
        Include ambient genes as a group in the plot/output dictionary. If `True`,
        `adata.var` must have a boolean column called 'ambient' labeling ambient genes.

    Returns
    -------
    markers : dict
        Dictionary of DEGs group names as keys and lists of genes as values
    """
    if groups is None:
        groups = adata.uns[uns_key]["names"].dtype.names
    else:
        assert set(groups).issubset(adata.uns[uns_key]["names"].dtype.names), "All given 'groups' must be present in adata.uns[uns_key]"

    # get markers manually
    markers = {}
    for clu in groups:
        markers[clu] = [
            adata.uns[uns_key]["names"][x][clu] for x in range(n_genes)
        ]
    # append ambient genes
    if ambient:
        if "ambient" in adata.var:
            markers["ambient"] = adata.var_names[adata.var["ambient"]].tolist()
        else:
            print("No 'ambient' column detected, skipping ambient genes")

    # total and unique features on plot
    features = signature_dict_values(signatures_dict=markers, unique=False)
    unique_features = signature_dict_values(signatures_dict=markers, unique=True)

    print(
        "Detected {} total features and {} unique features across {} groups".format(
            len(features), len(unique_features), len(groups)
        )
    )

    return markers


def flip_signature_dict(signatures_dict):
    """
    "Flip" dictionary of signatures where keys are signature names and values are lists
    of features, returning a dictionary where keys are individual features and values
    are signature names

    Parameters
    ----------
    signatures_dict : dict
        dictionary where keys are signature names and values are lists of features

    Returns
    -------
    signatures_dict_flipped : dict
        dictionary where keys are features and values are signature names
    """
    signatures_dict_flipped = {}
    for key, value in signatures_dict.items():
        for string in value:
            signatures_dict_flipped.setdefault(string, []).append(key)
    return signatures_dict_flipped


def signature_dict_values(signatures_dict, unique=True):
    """
    Extract features from dictionary of signatures where keys are signature names and
    values are lists of features, returning a single list of features

    Parameters
    ----------
    signatures_dict : dict
        dictionary where keys are signature names and values are lists of features
    unique : bool, optional (default=`True`)
        get only unique features across all `signatures_dict.values()`

    Returns
    -------
    dict_values : list
    """
    dict_values = [item for sublist in signatures_dict.values() for item in sublist]
    if unique:
        dict_values = list(set(dict_values))
    return dict_values


def filter_signatures_with_var_names(signatures_dict, adata):
    """
    Filter lists of genes in `signatures_dict` to include genes in `adata.var_names`
    """
    for key in signatures_dict.keys():
        signatures_dict[key] = list(
            set(signatures_dict[key]).intersection(set(adata.var_names))
        )
    return signatures_dict


def fetch_decoupler_resources(
    resources=["msigdb", "panglaodb", "progeny", "collectri", "liana"], genome="human"
):
    """
    Retrieve prior-knowledge networks from OmniPath for use with `decoupler` pathway
    analysis methods
    * MSigDB: biological pathways from HALLMARK (ORA)
    * PanglaoDB: cell-type and cell-state for scRNA labeling (ORA)
    * PROGENy: canonical signaling pathways (MLM)
    * CollecTRI: transcription factor regulon networks (ULM)
    * LIANA: ligand-receptor interactions (ULM)

    Parameters
    ----------
    resources : list, optional (default=["msigdb","panglaodb","progeny","collectri","liana"])
        List of resources to fetch. Default all; remove networks not desired.
    genome : str literal, optional (default="human")
        One of "human" or "mouse" to determine which gene symbols to return in
        `genesymbol` or `target` columns of network dataframes

    Returns
    -------
    nets : dict
        Dictionary containing names of OmniPath networks (keys) and the corresponding
        dataframes containing gene-pathway information (values).
    """
    assert genome.lower() in [
        "human",
        "mouse",
    ], "Please provide 'human' or 'mouse' for genome choice"
    for resource in resources:
        assert resource in [
            "msigdb",
            "panglaodb",
            "progeny",
            "collectri",
            "liana",
        ], f"Invalid resource: {resource}"
    # initiate output dict
    nets = {}

    if "msigdb" in resources:
        # Query Omnipath and get MSigDB
        print("Fetching MSigDB...", end=" ")
        msigdb = dc.get_resource("MSigDB", organism="human")
        # Filter by HALLMARK
        msigdb = msigdb.loc[
            msigdb["collection"].isin(["hallmark"])
        ]  # kegg_pathways, go_biological_process
        ## read in GO terms
        # go = dc.read_gmt(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../resources/c5.go.bp.v7.2.symbols.gmt"))
        # go.columns = ["geneset","genesymbol"]
        # go["collection"] = "GO"
        ## add GO to HALLMARK
        # msigdb = pd.concat([msigdb,go])
        # Remove duplicated entries
        msigdb = msigdb.loc[~msigdb.duplicated(["geneset", "genesymbol"])]
        # remove "HALLMARK_" from every term to shorten strings
        msigdb.geneset = msigdb.geneset.str.replace("HALLMARK_", "HM_")
        if genome == "mouse":
            # mouse symbols
            msigdb["genesymbol"] = msigdb["genesymbol"].map(human_to_mouse_simple)
        # add to dict
        nets["msigdb"] = msigdb
        print("network in nets['msigdb']")

    if "panglaodb" in resources:
        # Query Omnipath and get PanglaoDB
        print("Fetching PanglaoDB...", end=" ")
        panglaodb = dc.get_resource("PanglaoDB", organism="human")
        # Filter by canonical_marker and mouse
        panglaodb = panglaodb.loc[
            (panglaodb["mouse"] == True) & (panglaodb["canonical_marker"] == True)
        ]
        # Remove duplicated entries
        panglaodb = panglaodb[~panglaodb.duplicated(["cell_type", "genesymbol"])]
        if genome == "mouse":
            # mouse symbols
            panglaodb["genesymbol_mm"] = panglaodb["genesymbol"].map(
                human_to_mouse_simple
            )
        # add to dict
        nets["panglaodb"] = panglaodb
        print("network in nets['panglaodb']")

    if "collectri" in resources:
        # Query Omnipath and get CollecTRI
        collectri = dc.get_collectri(organism=genome, split_complexes=False)
        # add to dict
        nets["collectri"] = collectri

    if "progeny" in resources:
        # Query Omnipath and get PROGENy
        print("Fetching PROGENy...", end=" ")
        progeny = dc.get_progeny(organism="human", top=300)
        if genome == "mouse":
            # mouse genes
            progeny["target"] = progeny["target"].map(human_to_mouse_simple)
        # add to dict
        nets["progeny"] = progeny
        print("network in nets['progeny']")

    if "liana" in resources:
        # import LIANA ligand-receptor database
        print("Fetching LIANA...", end=" ")
        liana_lr = ln.resource.select_resource()
        liana_lr = ln.resource.explode_complexes(liana_lr)
        # create two new DataFrames, each containing one of the pairs of columns to be
        # concatenated
        df1 = liana_lr[["interaction", "ligand"]]
        df2 = liana_lr[["interaction", "receptor"]]
        # Rename the columns in each new DataFrame
        df1.columns = ["interaction", "genes"]
        df2.columns = ["interaction", "genes"]
        # Concatenate the two new DataFrames
        liana_lr = pd.concat([df1, df2], axis=0)
        liana_lr["weight"] = 1
        # Find duplicated rows
        duplicates = liana_lr.duplicated()
        # Remove duplicated rows
        liana_lr = liana_lr[~duplicates]
        if genome == "mouse":
            # mouse symbols
            liana_lr["genes"] = liana_lr["genes"].map(human_to_mouse_simple)
        # add to dict
        nets["liana"] = liana_lr
        print("network in nets['liana']")

    print("Done!")
    return nets
