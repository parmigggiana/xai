import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Config grafica
sns.set_theme(style="whitegrid")

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_PATTERNS = {
    "CHAOS_CT": re.compile(r"CHAOS_CT", re.IGNORECASE),
    "CHAOS_MR": re.compile(r"CHAOS_MR", re.IGNORECASE),
    "MMWHS_CT": re.compile(r"MMWHS_CT", re.IGNORECASE),
    "MMWHS_MR": re.compile(r"MMWHS_MR", re.IGNORECASE),
}

# Identifica modelli fine-tuned e composite
# Heuristic: *_2d_finetuned considered finetuned; names containing 'composite' considered composite.


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File non trovato: {path}")
    df = pd.read_csv(path)
    # Filtra solo split test per il confronto principale
    # Manteniamo comunque train/val per eventuali future estensioni
    return df


def classify_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_finetuned"] = df["model"].str.contains("2d_finetuned", case=False, na=False)
    df["is_composite"] = df["model"].str.contains("composite", case=False, na=False)
    df["is_baseline"] = df["model"].str.contains("2d_baseline", case=False, na=False)
    # Escludi adaptation per questa analisi
    df = df[~df["model"].str.contains("adaptation", case=False, na=False)]
    return df


def extract_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Determina il dataset logico (CHAOS_CT, CHAOS_MR, MMWHS_CT, MMWHS_MR) anche per nomi composite.

    Regole aggiuntive per i composite:
      - ^(CHAOS|MMWHS)_composite_at_(CT|MR)  -> {prefix}_{suffix}
      - ^(CT|MR)_composite_at_(CHAOS|MMWHS)  -> {second}_{first}
    """
    forward_re = re.compile(r"^(CHAOS|MMWHS)_composite_at_(CT|MR)", re.IGNORECASE)
    reverse_re = re.compile(r"^(CT|MR)_composite_at_(CHAOS|MMWHS)", re.IGNORECASE)
    mapping = {}
    for m in df["model"].unique():
        ds = None
        # Primo tentativo: pattern diretti (baseline/finetuned contengono già CHAOS_CT ecc.)
        for key, pattern in DATASET_PATTERNS.items():
            if pattern.search(m):
                ds = key
                break
        if ds is None:
            fwd = forward_re.search(m)
            if fwd:
                p1 = fwd.group(1).upper()  # CHAOS / MMWHS
                p2 = fwd.group(2).upper()  # CT / MR
                ds = f"{p1}_{p2}"
        if ds is None:
            rev = reverse_re.search(m)
            if rev:
                p1 = rev.group(1).upper()  # CT / MR
                p2 = rev.group(2).upper()  # CHAOS / MMWHS
                ds = f"{p2.upper()}_{p1.upper()}"
        mapping[m] = ds
    out = df.copy()
    out["dataset"] = out["model"].map(mapping)
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = classify_rows(df)
    df = extract_dataset(df)
    # Manteniamo tutte le righe (train/val/test) per flessibilità; il filtro test sarà applicato dove serve
    df = df[df["dataset"].notna()]
    # Normalizza alpha (potrebbe essere stringa o float)
    # Già float nel CSV attuale.
    return df


def plot_dataset_effect(df: pd.DataFrame, dataset: str):
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        print(f"[warn] Nessun dato per dataset {dataset}")
        return
    # Aggrega per alpha e tipo (finetuned/composite)
    sub = sub.copy()
    sub["tipo"] = sub.apply(
        lambda r: (
            "finetuned"
            if r.is_finetuned
            else ("composite" if r.is_composite else "altro")
        ),
        axis=1,
    )
    sub = sub[sub["tipo"].isin(["finetuned", "composite"])]
    if sub.empty:
        print(f"[warn] Nessun dato finetuned/composite per {dataset}")
        return

    # Calcola media dice per (alpha, tipo)
    grouped = sub.groupby(["alpha", "tipo"], as_index=False)["dice"].mean()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=grouped, x="alpha", y="dice", hue="tipo")
    plt.title(f"Effetto di alpha su {dataset} (bar)")
    plt.ylabel("Dice (media)")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"effect_alpha_{dataset}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Salvato: {out_path}")


def plot_overall_mean(df: pd.DataFrame):
    # Media su tutti i dataset per alpha e tipo
    df = df.copy()
    df["tipo"] = df.apply(
        lambda r: (
            "finetuned"
            if r.is_finetuned
            else ("composite" if r.is_composite else "altro")
        ),
        axis=1,
    )
    core = df[df["tipo"].isin(["finetuned", "composite"])]
    if core.empty:
        print("[warn] Nessun dato per il grafico complessivo")
        return
    grouped = core.groupby(["alpha", "tipo"], as_index=False)["dice"].mean()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=grouped, x="alpha", y="dice", hue="tipo")
    plt.title("Effetto di alpha (media su tutti i dataset) - bar")
    plt.ylabel("Dice (media)")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "effect_alpha_overall.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Salvato: {out_path}")


def _directional_plot(
    df: pd.DataFrame,
    dataset_label: str,
    composite_model: str,
    reverse_composite_model: str,
    baseline_model: str,
    finetuned_model: str,
):
    """Crea UNA figura con due subplot (forward / reverse) per il dataset.
    Ogni subplot: linea composite vs linee orizzontali baseline & finetuned.
    """
    tdf = df[df["split"] == "test"].copy()
    base_val = tdf.loc[tdf["model"] == baseline_model, "dice"].mean()
    ft_val = tdf.loc[tdf["model"] == finetuned_model, "dice"].mean()

    forward_df = (
        tdf[tdf["model"] == composite_model][["alpha", "dice"]]
        .copy()
        .sort_values("alpha")
    )
    reverse_df = (
        tdf[tdf["model"] == reverse_composite_model][["alpha", "dice"]]
        .copy()
        .sort_values("alpha")
    )
    if forward_df.empty and reverse_df.empty:
        print(f"[warn] Nessun dato forward/reverse per {dataset_label}")
        return

    # Range alpha unificato per disegnare eventuale griglia coerente
    all_alpha = pd.Series(
        pd.concat([forward_df["alpha"], reverse_df["alpha"]], ignore_index=True)
        .dropna()
        .unique()
    ).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    plt.suptitle(f"{dataset_label}: forward & reverse")

    def draw(ax, comp_df: pd.DataFrame, title: str):
        if comp_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xlabel("alpha")
            ax.set_ylabel("Dice (test)")
            return
        sns.barplot(
            data=comp_df,
            x="alpha",
            y="dice",
            ax=ax,
            color="#1f77b4",
        )
        # Manual legend entry
        ax.legend(
            handles=[
                plt.Line2D(
                    [], [], color="#1f77b4", marker="s", linestyle="", label="composite"
                )
            ]
        )
        if pd.notna(base_val):
            ax.axhline(base_val, color="#ff7f0e", linestyle="--", label="baseline")
        if pd.notna(ft_val):
            ax.axhline(ft_val, color="#2ca02c", linestyle=":", label="finetuned")
        ax.set_title(title)
        ax.set_xlabel("alpha")
        ax.set_ylabel("Dice (test)")
        # Garantisce che tutte le alphas siano mostrate (se discrete e poche)
        if len(all_alpha) <= 6 and not all_alpha.empty:
            ax.set_xticks(all_alpha.tolist())
        ax.legend()

    draw(axes[0], forward_df, composite_model)
    draw(axes[1], reverse_df, reverse_composite_model)
    # Rinomina titoli con etichette generiche richieste
    # root dataset es: CHAOS_CT -> CHAOS
    root_ds = dataset_label.split("_")[0]
    axes[0].set_title(f"{root_ds} composite")
    # modality è la seconda parte (es: CHAOS_CT -> CT)
    parts = dataset_label.split("_")
    modality = parts[1] if len(parts) > 1 else "CT/MR"
    axes[1].set_title(f"{modality} composite")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_name = f"dirplot_{dataset_label}_combined.png".replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")


def plot_all_directional(df: pd.DataFrame):
    """Definisce mapping dei modelli per ogni dataset e produce i grafici richiesti.
    Dataset CHAOS_CT:
        - CHAOS_composite_at_CT vs CHAOS_CT_2d_baseline & CHAOS_CT_2d_finetuned
        - CT_composite_at_CHAOS vs baseline/finetuned
    Simile per gli altri dataset.
    """
    specs = [
        {
            "dataset_label": "CHAOS_CT",
            "composite_model": "CHAOS_composite_at_CT",
            "reverse_composite_model": "CT_composite_at_CHAOS",
            "baseline_model": "CHAOS_CT_2d_baseline",
            "finetuned_model": "CHAOS_CT_2d_finetuned",
        },
        {
            "dataset_label": "CHAOS_MR",
            "composite_model": "CHAOS_composite_at_MR",
            "reverse_composite_model": "MR_composite_at_CHAOS",
            "baseline_model": "CHAOS_MR_2d_baseline",
            "finetuned_model": "CHAOS_MR_2d_finetuned",
        },
        {
            "dataset_label": "MMWHS_CT",
            "composite_model": "MMWHS_composite_at_CT",
            "reverse_composite_model": "CT_composite_at_MMWHS",
            "baseline_model": "MMWHS_CT_2d_baseline",
            "finetuned_model": "MMWHS_CT_2d_finetuned",
        },
        {
            "dataset_label": "MMWHS_MR",
            "composite_model": "MMWHS_composite_at_MR",
            "reverse_composite_model": "MR_composite_at_MMWHS",
            "baseline_model": "MMWHS_MR_2d_baseline",
            "finetuned_model": "MMWHS_MR_2d_finetuned",
        },
    ]
    for spec in specs:
        _directional_plot(df, **spec)


def plot_mean_models_vs_alpha(df: pd.DataFrame):
    """Grafico della media dei modelli composite sul test rispetto ad alpha.
    Linee orizzontali per media baseline e media finetuned.
    """
    tdf = df[df["split"] == "test"].copy()
    composites = tdf[tdf["is_composite"] & tdf["alpha"].notna()]
    if composites.empty:
        print("[warn] Nessun modello composite con alpha definito per media")
        return
    grouped = (
        composites.groupby("alpha", as_index=False)["dice"].mean().sort_values("alpha")
    )
    base_mean = tdf[tdf["is_baseline"]]["dice"].mean()
    ft_mean = tdf[tdf["is_finetuned"]]["dice"].mean()
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=grouped, x="alpha", y="dice", color="#1f77b4", label="composite mean"
    )
    if pd.notna(base_mean):
        plt.axhline(base_mean, color="#ff7f0e", linestyle="--", label="baseline mean")
    if pd.notna(ft_mean):
        plt.axhline(ft_mean, color="#2ca02c", linestyle=":", label="finetuned mean")
    plt.title("Average Performances (test)")
    plt.ylabel("Dice")
    plt.xlabel("Alpha")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "mean_models_vs_alpha.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Salvato: {out_path}")


def plot_composites_column(df: pd.DataFrame):
    """Crea una singola figura con 4 subplot (colonna):
    CHAOS composite, MMWHS composite, CT composite, MR composite.
    Ogni subplot mostra la media (test) del dice dei modelli composite del gruppo per alpha.
    Assunzioni:
      - Gruppi: prefix 'CHAOS_composite_at_', 'MMWHS_composite_at_', 'CT_composite_at_', 'MR_composite_at_'
      - Media baseline/finetuned: media dei modelli baseline/finetuned corrispondenti (per CHAOS: entrambi CT/MR; per MMWHS idem; per CT: media di CT baselines (CHAOS_CT_2d_baseline, MMWHS_CT_2d_baseline); MR analogo).
    """
    tdf = df[df["split"] == "test"].copy()
    groups = [
        ("CHAOS composite", r"^CHAOS_composite_at_"),
        ("MMWHS composite", r"^MMWHS_composite_at_"),
        ("CT composite", r"^CT_composite_at_"),
        ("MR composite", r"^MR_composite_at_"),
    ]
    fig, axes = plt.subplots(len(groups), 1, figsize=(6, 14), sharex=True)
    all_alpha = sorted(tdf["alpha"].dropna().unique())

    def mean_curve(pattern: str):
        subset = tdf[
            tdf["model"].str.contains(pattern, regex=True, na=False)
            & tdf["is_composite"]
        ]
        if subset.empty:
            return pd.DataFrame(columns=["alpha", "dice"])
        return (
            subset.groupby("alpha", as_index=False)["dice"].mean().sort_values("alpha")
        )

    # Pre-calcola baseline / finetuned mapping per facilitare
    baselines = tdf[tdf["is_baseline"]]
    finetuned = tdf[tdf["is_finetuned"]]

    def mean_for_models(models_regex_list, frame):
        if frame.empty:
            return float("nan")
        mask = False
        for rgx in models_regex_list:
            mask = mask | frame["model"].str.contains(rgx, regex=True, na=False)
        vals = frame[mask]["dice"]
        return vals.mean() if not vals.empty else float("nan")

    baseline_map = {
        "CHAOS composite": mean_for_models([r"^CHAOS_.._2d_baseline"], baselines),
        "MMWHS composite": mean_for_models([r"^MMWHS_.._2d_baseline"], baselines),
        "CT composite": mean_for_models(
            [r"^CHAOS_CT_2d_baseline", r"^MMWHS_CT_2d_baseline"], baselines
        ),
        "MR composite": mean_for_models(
            [r"^CHAOS_MR_2d_baseline", r"^MMWHS_MR_2d_baseline"], baselines
        ),
    }
    finetuned_map = {
        "CHAOS composite": mean_for_models([r"^CHAOS_.._2d_finetuned"], finetuned),
        "MMWHS composite": mean_for_models([r"^MMWHS_.._2d_finetuned"], finetuned),
        "CT composite": mean_for_models(
            [r"^CHAOS_CT_2d_finetuned", r"^MMWHS_CT_2d_finetuned"], finetuned
        ),
        "MR composite": mean_for_models(
            [r"^CHAOS_MR_2d_finetuned", r"^MMWHS_MR_2d_finetuned"], finetuned
        ),
    }

    for ax, (label, pattern) in zip(axes, groups):
        curve = mean_curve(pattern)
        if curve.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            sns.barplot(
                data=curve,
                x="alpha",
                y="dice",
                ax=ax,
                color="#1f77b4",
            )
            ax.legend(
                handles=[
                    plt.Line2D(
                        [],
                        [],
                        color="#1f77b4",
                        marker="s",
                        linestyle="",
                        label="composite mean",
                    )
                ],
                fontsize=8,
            )
        bval = baseline_map.get(label, float("nan"))
        fval = finetuned_map.get(label, float("nan"))
        if pd.notna(bval):
            ax.axhline(bval, color="#ff7f0e", linestyle="--", label="baseline mean")
        if pd.notna(fval):
            ax.axhline(fval, color="#2ca02c", linestyle=":", label="finetuned mean")
        ax.set_title(label)
        ax.set_ylabel("Dice")
        if len(all_alpha) <= 6:
            ax.set_xticks(all_alpha)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("alpha")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "composites_column_overview.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")


def main():
    df = load_data(RESULTS_CSV)
    df = prepare(df)

    # Grafico media modelli composite vs alpha
    plot_mean_models_vs_alpha(df)
    # Figura 8 subplot (richiesta)
    plot_composites_8panel(df)
    # Barplot alpha=0.8
    plot_bar_alpha(df, target_alpha=0.8)
    # Adaptation mean vs alpha
    plot_adaptation_mean_vs_alpha()
    # Adaptation 4 panel
    plot_adaptation_4panel()


def plot_composites_8panel(df: pd.DataFrame):
    """Crea figura 2x4:
    Colonne:
      0: CHAOS composite ( @ CT / @ MR )
      1: MMWHS composite ( @ CT / @ MR )
      2: CT composite ( @ CHAOS / @ MMWHS )
      3: MR composite ( @ CHAOS / @ MMWHS )
    Ogni subplot: linea composite (dice vs alpha) + linee orizzontali baseline & finetuned del dataset target.
    """
    tdf = df[df["split"] == "test"].copy()

    panels = [
        (
            "CHAOS_composite_at_CT",
            "CHAOS_CT_2d_baseline",
            "CHAOS_CT_2d_finetuned",
            "CHAOS composite @ CT",
        ),
        (
            "CHAOS_composite_at_MR",
            "CHAOS_MR_2d_baseline",
            "CHAOS_MR_2d_finetuned",
            "CHAOS composite @ MR",
        ),
        (
            "MMWHS_composite_at_CT",
            "MMWHS_CT_2d_baseline",
            "MMWHS_CT_2d_finetuned",
            "MMWHS composite @ CT",
        ),
        (
            "MMWHS_composite_at_MR",
            "MMWHS_MR_2d_baseline",
            "MMWHS_MR_2d_finetuned",
            "MMWHS composite @ MR",
        ),
        (
            "CT_composite_at_CHAOS",
            "CHAOS_CT_2d_baseline",
            "CHAOS_CT_2d_finetuned",
            "CT composite @ CHAOS",
        ),
        (
            "CT_composite_at_MMWHS",
            "MMWHS_CT_2d_baseline",
            "MMWHS_CT_2d_finetuned",
            "CT composite @ MMWHS",
        ),
        (
            "MR_composite_at_CHAOS",
            "CHAOS_MR_2d_baseline",
            "CHAOS_MR_2d_finetuned",
            "MR composite @ CHAOS",
        ),
        (
            "MR_composite_at_MMWHS",
            "MMWHS_MR_2d_baseline",
            "MMWHS_MR_2d_finetuned",
            "MR composite @ MMWHS",
        ),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 8), sharey=True)

    legend_handles = None
    for idx, (model, base, ft, title) in enumerate(panels):
        row = idx % 2
        col = idx // 2
        ax = axes[row, col]
        comp = tdf[tdf["model"] == model].copy()
        if comp.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
        else:
            comp = comp[["alpha", "dice"]].sort_values("alpha")
            sns.barplot(data=comp, x="alpha", y="dice", ax=ax, color="#1f77b4")
            bval = tdf.loc[tdf["model"] == base, "dice"].mean()
            fval = tdf.loc[tdf["model"] == ft, "dice"].mean()
            baseline_line = ft_line = None
            if pd.notna(bval):
                baseline_line = ax.axhline(bval, color="#ff7f0e", linestyle="--")
            if pd.notna(fval):
                ft_line = ax.axhline(fval, color="#2ca02c", linestyle=":")
            if legend_handles is None:
                handles_tmp = []
                handles_tmp.append(
                    plt.Line2D(
                        [],
                        [],
                        color="#1f77b4",
                        marker="s",
                        linestyle="",
                        label="composite",
                    )
                )
                if baseline_line is not None:
                    handles_tmp.append(baseline_line)
                if ft_line is not None:
                    handles_tmp.append(ft_line)
                legend_handles = handles_tmp
            # Rimuovi eventuale legenda automatica creata da seaborn
            leg = ax.get_legend()
            if leg:
                leg.remove()
            ax.set_title(title)
        if row == 1:
            ax.set_xlabel("alpha")
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel("Dice (test)")
        else:
            ax.set_ylabel("")

    # Colonne label sovra-titolo
    # col_titles = ["CHAOS composite", "MMWHS composite", "CT composite", "MR composite"]
    # for c, ct in enumerate(col_titles):
    #     fig.text(
    #         (c + 0.5) / 4.0,
    #         0.995,
    #         ct,
    #         ha="center",
    #         va="top",
    #         fontsize=11,
    #         fontweight="bold",
    #     )

    # Costruisci legenda unica
    if legend_handles:
        labels = ["composite", "baseline", "finetuned"][: len(legend_handles)]
        fig.legend(
            legend_handles,
            labels,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "composites_8panel.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {out_path}")

    # (Opzionale) salva tabella riassuntiva
    summary = df.copy()
    summary["tipo"] = summary.apply(
        lambda r: (
            "finetuned"
            if r.is_finetuned
            else ("composite" if r.is_composite else "altro")
        ),
        axis=1,
    )
    summary = summary[summary["tipo"].isin(["finetuned", "composite"])]
    table = (
        summary.groupby(["dataset", "alpha", "tipo"], as_index=False)["dice"]
        .mean()
        .sort_values(["dataset", "tipo", "alpha"])
    )
    summary_path = os.path.join(OUTPUT_DIR, "alpha_effect_summary.csv")
    table.to_csv(summary_path, index=False)
    print(f"Salvato: {summary_path}")


def plot_bar_alpha(df: pd.DataFrame, target_alpha: float = 0.8):
    """Barplot per i 4 dataset (CHAOS_CT, CHAOS_MR, MMWHS_CT, MMWHS_MR) usando solo alpha target per i composite
    Barre: baseline, composite_forward, composite_reverse, finetuned.
    """
    tdf = df[df["split"] == "test"].copy()
    datasets = ["CHAOS_CT", "CHAOS_MR", "MMWHS_CT", "MMWHS_MR"]
    specs = {
        "CHAOS_CT": {
            "baseline": "CHAOS_CT_2d_baseline",
            "finetuned": "CHAOS_CT_2d_finetuned",
            "composite_fwd": "CHAOS_composite_at_CT",
            "composite_rev": "CT_composite_at_CHAOS",
        },
        "CHAOS_MR": {
            "baseline": "CHAOS_MR_2d_baseline",
            "finetuned": "CHAOS_MR_2d_finetuned",
            "composite_fwd": "CHAOS_composite_at_MR",
            "composite_rev": "MR_composite_at_CHAOS",
        },
        "MMWHS_CT": {
            "baseline": "MMWHS_CT_2d_baseline",
            "finetuned": "MMWHS_CT_2d_finetuned",
            "composite_fwd": "MMWHS_composite_at_CT",
            "composite_rev": "CT_composite_at_MMWHS",
        },
        "MMWHS_MR": {
            "baseline": "MMWHS_MR_2d_baseline",
            "finetuned": "MMWHS_MR_2d_finetuned",
            "composite_fwd": "MMWHS_composite_at_MR",
            "composite_rev": "MR_composite_at_MMWHS",
        },
    }
    rows = []
    for ds in datasets:
        mapping = specs[ds]
        base_val = tdf.loc[tdf["model"] == mapping["baseline"], "dice"].mean()
        ft_val = tdf.loc[tdf["model"] == mapping["finetuned"], "dice"].mean()
        # imaging technique composite (dataset_composite_at_CT/MR)
        img_val = tdf.loc[
            (tdf["model"] == mapping["composite_fwd"]) & (tdf["alpha"] == target_alpha),
            "dice",
        ].mean()
        # body part composite (CT/MR_composite_at_dataset)
        body_val = tdf.loc[
            (tdf["model"] == mapping["composite_rev"]) & (tdf["alpha"] == target_alpha),
            "dice",
        ].mean()
        rows.extend(
            [
                {"dataset": ds, "model_type": "baseline", "dice": base_val},
                {"dataset": ds, "model_type": "body_part_composite", "dice": body_val},
                {
                    "dataset": ds,
                    "model_type": "imaging_technique_composite",
                    "dice": img_val,
                },
                {"dataset": ds, "model_type": "finetuned", "dice": ft_val},
            ]
        )
    plot_df = pd.DataFrame(rows)
    # Ordine modelli
    order_models = [
        "baseline",
        "body_part_composite",
        "imaging_technique_composite",
        "finetuned",
    ]
    plot_df["model_type"] = pd.Categorical(
        plot_df["model_type"], categories=order_models, ordered=True
    )
    # Pretty names for legend
    pretty_map = {
        "baseline": "Baseline",
        "body_part_composite": "Body-part composite",
        "imaging_technique_composite": "Imaging-tech composite",
        "finetuned": "Finetuned",
    }
    plot_df["model_pretty"] = plot_df["model_type"].map(pretty_map)
    plot_df["model_pretty"] = pd.Categorical(
        plot_df["model_pretty"],
        categories=[pretty_map[m] for m in order_models],
        ordered=True,
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="dataset", y="dice", hue="model_pretty")
    # plt.title(f'Confronto modelli (alpha={target_alpha})')
    plt.ylabel("Dice (test)")
    plt.xlabel("Dataset")
    # Legend outside on the right
    leg = plt.legend(
        title="Model", loc="center left", bbox_to_anchor=(0, 0.4), borderaxespad=0
    )
    for txt in leg.get_texts():
        txt.set_fontsize(9)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "bar_overall.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Salvato: {out_path}")


def plot_adaptation_mean_vs_alpha():
    """Mean dice vs alpha for adaptation models (part=2) on test split.
    Separate lines for weight_balancing True/False and overall mean.
    """
    raw = pd.read_csv(RESULTS_CSV)
    adp = raw[
        (raw.get("part") == 2)
        & (raw["model"].str.contains("adaptation", na=False))
        & (raw["split"] == "test")
    ]
    if adp.empty:
        print("[warn] Nessun dato adaptation per il grafico mean vs alpha")
        return
    # Normalizza alpha (già float) e weight_balancing
    # Alcune colonne weight_balancing possono essere stringhe 'True'/'False'
    adp = adp.copy()
    # Convert dice to numeric (in case of strings) and drop NaN
    adp["dice"] = pd.to_numeric(adp["dice"], errors="coerce")
    adp = adp.dropna(subset=["alpha", "dice"])
    # weight balancing normalization
    wb = adp["weight_balancing"].astype(str).str.lower()
    adp["wb"] = wb.map({"true": "wb=True", "false": "wb=False"}).fillna("wb=NA")
    # Grouped means
    grouped_wb = adp.groupby(["alpha", "wb"], as_index=False)["dice"].mean()
    overall = adp.groupby("alpha", as_index=False)["dice"].mean()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=grouped_wb, x="alpha", y="dice", hue="wb")
    if not overall.empty:
        plt.plot(
            overall["alpha"],
            overall["dice"],
            color="black",
            marker="o",
            linestyle="-",
            label="overall",
        )
    plt.title("Adaptation models (bar)")
    plt.ylabel("Mean Dice (test)")
    plt.xlabel("Alpha")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "adaptation_mean_models_vs_alpha.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Salvato: {out_path}")


def plot_adaptation_4panel():
    """4 subplot figure for adaptation models:
    Each dataset in one subplot, two curves (wb=False, wb=True) + baseline line.
    """
    raw = pd.read_csv(RESULTS_CSV)
    adp = raw[
        (raw.get("part") == 2)
        & (raw["model"].str.contains("adaptation", na=False))
        & (raw["split"] == "test")
    ].copy()
    if adp.empty:
        print("[warn] Nessun dato adaptation per 4-panel")
        return
    adp["dice"] = pd.to_numeric(adp["dice"], errors="coerce")
    adp = adp.dropna(subset=["dice", "alpha"])
    adp["wb_flag"] = (
        adp["weight_balancing"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    datasets = ["CHAOS_CT", "CHAOS_MR", "MMWHS_CT", "MMWHS_MR"]
    baselines = raw[(raw["split"] == "test") & raw["model"].str.contains("2d_baseline")]
    baseline_vals = {
        m.replace("_2d_baseline", ""): baselines[baselines["model"] == m]["dice"].mean()
        for m in baselines["model"].unique()
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=True)
    colors = {False: "#1f77b4", True: "#d62728"}
    for col, ds in enumerate(datasets):
        ax = axes[col]
        for wb_flag in [False, True]:
            subset = adp[
                (adp["model"] == f"{ds}_adaptation") & (adp["wb_flag"] == wb_flag)
            ]
            if subset.empty:
                continue
            # accumulate; handled after loop
        subset_all = adp[(adp["model"] == f"{ds}_adaptation")]
        if not subset_all.empty:
            grouped_ds = subset_all.groupby(["alpha", "wb_flag"], as_index=False)[
                "dice"
            ].mean()
            sns.barplot(
                data=grouped_ds,
                x="alpha",
                y="dice",
                hue="wb_flag",
                ax=ax,
                palette={False: colors[False], True: colors[True]},
            )
            # Adjust legend entries per axis removed later
        bval = baseline_vals.get(ds, float("nan"))
        if pd.notna(bval):
            ax.axhline(bval, color="#ff7f0e", linestyle="--", label="baseline")
        ax.set_title(ds)
        if col == 0:
            ax.set_ylabel("Dice (test)")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Alpha")
    handles, labels = [], []
    for ax in axes:
        leg = ax.get_legend()
        if leg:
            h, lbls = leg.legend_handles, [t.get_text() for t in leg.texts]
            for hi, li in zip(h, lbls):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
            leg.remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "adaptation_4panel.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {out_path}")


if __name__ == "__main__":
    main()
