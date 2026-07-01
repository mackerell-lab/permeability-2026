from glob import glob
import pandas as pd
from scipy.stats import levene
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
import scikit_posthocs as sp
from pathlib import Path


def adapt_nested_cv_metrics(
    csvfile,
    outer_fold_col="outer_fold",
    inner_fold_col="inner_fold",
    split_col="split",
    n_inner=5,
):
    """
    Adapter for precomputed nested-CV regression metrics.

    """
    df = pd.read_csv(csvfile)
    df = df.copy()

    # --- matrix-style cv_cycle ---
    df["cv_cycle"] = (
        (df[outer_fold_col] - 1) * n_inner
        + (df[inner_fold_col] - 1)
    )

    if "basename" in df.columns:
        col = "basename"
        ext = ".pt"
    elif "model_file" in df.columns:
        col = "model_file"
        ext = ".rds"
    else:
        raise ValueError("Neither 'basename' nor 'model_file' found in dataframe")

    # Remove extension
    base = df[col].str.replace(ext, "", regex=False)

    # Remove the training / split / fold block
    # matches: _1_train_butina_fold1, _1_train_random_fold3, etc.
    base = base.str.replace(
        r"_\d+_train_[^_]+_fold\d+",
        "",
        regex=True,
    )

    df["method"] = base    

    # --- return: metrics untouched ---
    return df[
        [
            "cv_cycle",
            "method",
            split_col,
            "R2",
            "RMSE",
            "MAE",
            "PearsonR",
            "SpearmanRho",
        ]
    ]

def concat_metric_files(csv_files):
    
    files = []
    for path in csv_files:
        files.extend(sorted(glob(path)))
    if not files:
        raise ValueError(f"No CSV files matched pattern: {csv_files}")
    
    df_list = []
    for csv_file in files:
        df_tmp = adapt_nested_cv_metrics(csv_file)
        df_list.append(df_tmp)
    return pd.concat(df_list, ignore_index=True)

def variance_analysis_by_split(
    df,
    metric_cols,
    split_col="split",
    method_col="method",
):
    """
    Compute variance instability diagnostics across methods.

    Returns a tidy dataframe with:
      split | metric | levene_pvalue | max_over_min
    """

    records = []

    for split_name, df_split in df.groupby(split_col):
        for metric in metric_cols:
            # collect per-method metric lists
            groups = (
                df_split
                .groupby(method_col)[metric]
                .apply(list)
                .dropna()
            )

            # --- Levene p-value ---
            if len(groups) >= 2:
                _, pval = levene(*groups)
            else:
                pval = None

            # --- Variance ratio ---
            variances = (
                df_split
                .groupby(method_col)[metric]
                .var()
                .dropna()
            )

            if len(variances) >= 2 and variances.min() > 0:
                max_over_min = variances.max() / variances.min()
            else:
                max_over_min = None

            records.append({
                "split": split_name,
                "metric": metric,
                "levene_pvalue": pval,
                "max_over_min": max_over_min,
            })

    return pd.DataFrame(records)

def extract_common_prefix(strings, sep="_"):
    """
    Extract common prefix across strings split by `sep`.
    """
    if len(strings) < 2:
        return ""

    split_strings = [s.split(sep) for s in strings]
    common = []

    for tokens in zip(*split_strings):
        if all(t == tokens[0] for t in tokens):
            common.append(tokens[0])
        else:
            break

    return sep.join(common)


def make_boxplots_parametric(
    df,
    metric_ls,
    method_col="method",
    subject_col="cv_cycle",
    prefix="spliting",
    outdir=None,
    common_prefix=None,
    line_breaks=True,
):
    """
    Parametric boxplots using repeated-measures ANOVA with automatic
    redundancy removal in method labels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        - method_col (e.g. model / architecture name)
        - subject_col (e.g. cv_cycle)
        - metrics in metric_ls
    metric_ls : list[str]
        Metrics to plot
    method_col : str
        Column indicating method / model
    subject_col : str
        Repeated-measures subject identifier

    Returns
    -------
    None
    """

    sns.set_context("notebook")
    sns.set_style("whitegrid")

    n_metrics = len(metric_ls)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(6 * n_metrics, 6), sharey=False
    )

    if n_metrics == 1:
        axes = [axes]

    # --- Detect redundant text in method names ---
    methods = df[method_col].unique().tolist()

    if common_prefix is None:
        common_prefix = extract_common_prefix(methods)

    # Clean title-friendly version
    # title_prefix = (
    #     (common_prefix.replace("_", " ").replace("-", " ").strip() + "\n" + prefix)
    #     if common_prefix
    #     else None
    # )
    title_prefix = prefix

    for ax, metric in zip(axes, metric_ls):

        # --- Repeated-measures ANOVA ---
        try:
            model = AnovaRM(
                data=df,
                depvar=metric,
                subject=subject_col,
                within=[method_col],
                aggregate_func="mean",  # fixes duplicate obs warning
            ).fit()
            p_value = model.anova_table["Pr > F"].iloc[0]
            p_text = f"ANOVA p={p_value:.1e}"
        except Exception as e:
            print(f"Warning: ANOVA failed for {metric}: {e}")
            p_text = "ANOVA failed"

        # --- Boxplot ---
        sns.boxplot(
            data=df,
            x=method_col,
            y=metric,
            hue=method_col,
            dodge=False,
            palette="Set2",
            ax=ax,
            legend=False,
        )

        ax.set_ylim([0.0,1.0])
        # ax.set_ylim([0.0,4.0])

        if title_prefix:
            fig.suptitle(
                title_prefix.replace("_", " ").replace("-", " "),
                fontsize=18 #,
                # y=1.02
            )

        # --- Per-axis title ---
        ax.set_title(f"{p_text}", fontsize=14)
        # ax.set_title(f"{metric}\n{p_text}", fontsize=14)

        # --- Shorten x tick labels ---
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        short_labels = []

        for label in x_labels:
            if common_prefix and common_prefix in label:
                short = label.replace(common_prefix, "")
                short = short.strip("_")
                if "arch" in short:
                    short = short.replace("_arch", "")
                if "-including-protomers" in short:
                    short = short.replace("-including-protomers", "explicit")
                if "_" in short and line_breaks:
                    short = short.rsplit("_", 1)[0] + "\n" + short.rsplit("_", 1)[1]
            else:
                short = label
            short_labels.append(short)


        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=90, ha="center", fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
        ax.set_xlabel("")
        # ax.set_ylabel(metric, fontsize=16)
 

    plt.tight_layout()
    # plt.grid(None)
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / ("parameteric_box_plots_" + common_prefix.replace("-", "_").strip() + "_" + prefix + ".png"), dpi=300)
    # plt.show()

def make_boxplots_nonparametric(
    df,
    metric_ls,
    method_col="method",
    subject_col="cv_cycle",
    prefix="spliting",
    outdir=None,
    common_prefix=None,
    line_breaks=True,
):
    """
    Non-parametric boxplots using Friedman test with automatic
    redundancy removal in method labels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        - method_col (e.g. model / architecture name)
        - subject_col (e.g. cv_cycle)
        - metrics in metric_ls
    metric_ls : list[str]
        Metrics to plot
    method_col : str
        Column indicating method / model
    subject_col : str
        Repeated-measures subject identifier

    Returns
    -------
    None
    """

    sns.set_context("notebook")
    sns.set_style("whitegrid")

    n_metrics = len(metric_ls)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(6 * n_metrics, 6), sharey=False
    )

    if n_metrics == 1:
        axes = [axes]

    # --- Detect redundant text in method names ---
    methods = df[method_col].unique().tolist()
    if common_prefix is None:
        common_prefix = extract_common_prefix(methods)

    # Clean title-friendly version
    # title_prefix = (
    #     (common_prefix.replace("_", " ").replace("-", " ").strip() + "\n" + prefix)
    #     if common_prefix
    #     else None
    # )
    title_prefix = prefix

    for ax, metric in zip(axes, metric_ls):

        # --- Friedman test ---
        try:
            friedman = pg.friedman(
                data=df,
                dv=metric,
                within=method_col,
                subject=subject_col
            )
            p_value = friedman["p-unc"].iloc[0]
            p_text = f"Friedman p={p_value:.1e}"
        except Exception as e:
            print(f"Warning: Friedman failed for {metric}: {e}")
            p_text = "Friedman failed"

        # --- Boxplot ---
        sns.boxplot(
            data=df,
            x=method_col,
            y=metric,
            hue=method_col,
            dodge=False,
            palette="Set2",
            ax=ax,
            legend=False,
        )

        # ax.set_ylim([0.0,1.0])
        metric_lower = metric.lower()

        if "pearsonr" in metric_lower:
            ax.set_ylim(0.0, 1.0)

        elif metric_lower in ["mae", "rmse"]:
            ax.set_ylim(0.0, 2.0)
            # ax.set_ylim([0.0,4.0])

        if title_prefix:
            fig.suptitle(
                title_prefix.replace("_", " ").replace("-", " "),
                fontsize=18) #,
                #y=1.02
            #)

        # --- Per-axis title ---
        ax.set_title(f"{p_text}", fontsize=14)

        # --- Shorten x tick labels ---
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        short_labels = []

        for label in x_labels:
            if common_prefix and common_prefix in label:
                short = label.replace(common_prefix, "")
                short = short.lstrip("_")
                if "arch" in short:
                    short = short.replace("_arch", "")
                if "-including-protomers" in short:
                    short = short.replace("-including-protomers", "explicit")
                if "_" in short and line_breaks:
                    short = short.rsplit("_", 1)[0] + "\n" + short.rsplit("_", 1)[1]
            else:
                short = label
            short_labels.append(short)


        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=90, ha="center", fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=16)

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    # plt.grid(None)
    plt.savefig(outdir / ("non_parameteric_box_plots_" + common_prefix.replace("-", "_").strip() + "_" + prefix + ".png"), dpi=300)
    # plt.show()


def make_sign_plots_nonparametric(
    df,
    metric_ls,
    method_col="method",
    subject_col="cv_cycle",
    prefix="spliting",
    outdir=None,
    common_prefix=None,
    line_breaks=True,
):
    """
    Non-parametric significance plots using Friedman + Conover post-hoc
    with automatic redundancy removal in method labels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - method_col
        - subject_col
        - metrics in metric_ls
    metric_ls : list[str]
        Metrics to plot
    method_col : str
        Column with model / method names
    subject_col : str
        Repeated-measures identifier
    prefix : str
        Extra text appended to the shared title
    """

    sns.set_context("notebook")
    sns.set_style("whitegrid")

    heatmap_args = {
        "linewidths": 0.25,
        "linecolor": "0.5",
        "clip_on": True,
        "square": True,
    }

    n_metrics = len(metric_ls)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(5 * n_metrics, 6), sharey=True
    )

    if n_metrics == 1:
        axes = [axes]

    # --- Detect redundant text in method names ---
    methods = df[method_col].unique().tolist()
    if common_prefix is None:
        common_prefix = extract_common_prefix(methods)

    # Clean title-friendly version
    title_prefix = (
        (common_prefix.replace("_", " ").replace("-", " ").strip() + "\n" + prefix)
        if common_prefix
        else None
    )
    

    fig.suptitle(title_prefix, fontsize=18) #, y=1.05)

    for ax, metric in zip(axes, metric_ls):

        # --- Pivot for Friedman / Conover ---
        pivot_df = df.pivot(
            index=subject_col,
            columns=method_col,
            values=metric
        )

        # --- Post-hoc Conover after Friedman ---
        pc = sp.posthoc_conover_friedman(
            pivot_df,
            p_adjust="holm"
        )

        # --- Significance plot ---
        sub_ax, _ = sp.sign_plot(
            pc,
            ax=ax,
            xticklabels=True,
            **heatmap_args,
        )

        # --- Shorten tick labels ---
        labels = [t.get_text() for t in sub_ax.get_xticklabels()]
        short_labels = []

        for label in labels:
            if common_prefix and common_prefix in label:
                short = label.replace(common_prefix, "")
                short = short.lstrip("_")
                if "arch" in short:
                    short = short.replace("_arch", "")
                if "-including-protomers" in short:
                    short = short.replace("-including-protomers", "explicit")
                if "_" in short and line_breaks:
                    short = short.rsplit("_", 1)[0] + "\n" + short.rsplit("_", 1)[1]
            else:
                short = label
            short_labels.append(short)


        sub_ax.set_xticklabels(short_labels, rotation=90, ha="center", fontsize=14)
        sub_ax.set_yticklabels(short_labels, fontsize=14)

        # --- Axis title ---
        sub_ax.set_title(metric, fontsize=16)

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
    # plt.tight_layout()
    plt.savefig(outdir / ("non_parameteric_sign_plots_" + common_prefix.replace("-", "_").strip() + "_" + prefix + ".png"), dpi=300)
    # plt.show()