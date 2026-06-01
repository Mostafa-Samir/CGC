import re
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import glob

import argparse

# === USER OPTIONS ===
# =====================

# ---------- Helpers ----------
def _canon(s: str) -> str:
    s = str(s).strip().lower()
    # normalize method names; accept "Two-Step" typo and spacing/hyphens
    s = re.sub(r"\btwo[\-\s]*steps?\b", "two-steps", s)
    s = re.sub(r"\bone[\-\s]*step\b", "one-step", s)
    s = s.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\(\s*", " (", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    return s

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Use 1st column as index; standardize to 'Mean'/'Std' labels."""
    first_col = df.columns[0]
    df2 = df.set_index(first_col)
    df2.index = [
        "Mean" if str(i).strip().lower() == "mean"
        else "Std" if str(i).strip().lower() in {"std","stdev","stddev"}
        else i
        for i in df2.index
    ]
    return df2

def _find_column(df: pd.DataFrame, desired: str) -> Optional[str]:
    want = _canon(desired)
    for col in df.columns:
        if _canon(col) == want:
            return col
    return None

def _get_re_cell(df: pd.DataFrame, method: str, channel: str, err: str) -> Tuple[float, float]:
    """
    Expect EXACT shape:
      '<Method> <Channel> RE (<err>)'
    where Method ∈ {'Two-Steps','One-Step'} (typo 'Two-Step' tolerated in CSV),
          Channel ∈ {'P','Q'},
          err ∈ {'Interpolation','Extrapolation'}.
    """
    label = f"{method} {channel} RE ({err})"
    col = _find_column(df, label)
    if col is None or "Mean" not in df.index or "Std" not in df.index:
        return (np.nan, np.nan)
    try:
        return float(df.loc["Mean", col]), float(df.loc["Std", col])
    except Exception:
        return (np.nan, np.nan)

def _pm(mean: float, std: float, d: int) -> str:
    if np.isnan(mean):
        return r"\textemdash{}"
    return f"{mean:.{d}f} $\\pm$ {std:.{d}f}"

# ---------- LaTeX ----------
def _build_re_table(rows: List[Tuple[str, Dict[Tuple[str,str,str], Tuple[float,float]]]], caption: str, label: str, decimals: int) -> str:
    methods = ["Two-Steps", "One-Step"]
    errs = ["Interpolation", "Extrapolation"]
    channels = ["P", "Q"]  # << ONLY P and Q

    per_method_cols = len(errs) * len(channels)  # 2 * 2 = 4
    n_data_cols = per_method_cols * len(methods) # 8

    H = []
    H.append(r"\begin{table}[ht]")
    H.append(r"\centering")
    H.append(r"\setlength{\tabcolsep}{5pt}")
    H.append(r"\renewcommand{\arraystretch}{1.1}")
    H.append(r"\scriptsize")
    H.append(r"\resizebox{\textwidth}{!}{%")
    H.append(r"\begin{tabular}{l" + "c"*n_data_cols + "}")
    H.append(r"\toprule")
    H.append(
        r"\multirow{3}{*}{\textbf{Sparsity}} & "
        + rf"\multicolumn{{{per_method_cols}}}{{c}}{{\textbf{{Two-Steps Method}}}} & "
        + rf"\multicolumn{{{per_method_cols}}}{{c}}{{\textbf{{One-Step Method}}}} \\"
    )
    # cmidrule ranges
    left_end = 1 + per_method_cols
    right_start = 2 + per_method_cols
    right_end = 1 + per_method_cols*2
    H.append(rf"\cmidrule(lr){{2-{left_end}}}\cmidrule(lr){{{right_start}-{right_end}}}")

    # row 2: Interp vs Extra for both methods
    H.append(
        r"& \multicolumn{2}{c}{Interpolation} & \multicolumn{2}{c}{Extrapolation} "
        r"& \multicolumn{2}{c}{Interpolation} & \multicolumn{2}{c}{Extrapolation} \\"
    )
    H.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")

    # row 3: channels P, Q repeated
    H.append(r"& $p$ & $q$ & $p$ & $q$ & $p$ & $q$ & $p$ & $q$ \\")
    H.append(r"\midrule")

    B = []
    for sparsity, vals in rows:
        cells = []
        for m in methods:
            for e in errs:
                for ch in channels:
                    mean,std = vals.get((m,e,ch),(np.nan,np.nan))
                    cells.append(_pm(mean,std,decimals))
        B.append(rf"{sparsity} & " + " & ".join(cells) + r" \\")

    F = []
    F.append(r"\bottomrule")
    F.append(r"\end{tabular}")
    F.append(r"}")
    F.append(caption)
    F.append(label)
    F.append(r"\end{table}")

    return "\n".join(H+B+F)

# ---------- Main ----------
def generate_re_pq_table(csv_paths: List[str], out_tex: str, caption: str, label: str, decimals: int = 3) -> str:
    rows = []
    for path in csv_paths:
        df_raw = pd.read_csv(path)
        df = _normalize_index(df_raw)

        # collect values for P, Q only
        vals: Dict[Tuple[str,str,str], Tuple[float,float]] = {}
        for method in ["Two-Steps", "One-Step"]:
            for err in ["Interpolation", "Extrapolation"]:
                for ch in ["P","Q"]:
                    vals[(method, err, ch)] = _get_re_cell(df, method, ch, err)

        sparsity = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r'(?:sparse|sparsity)[-_]?([0-9]*\.?[0-9]+)', sparsity, flags=re.IGNORECASE)
        if m: sparsity = m.group(1)
        rows.append((sparsity, vals))

    try:
        rows.sort(key=lambda x: float(x[0]))
    except Exception:
        pass

    tex = _build_re_table(rows, caption, label, decimals)
    with open(out_tex, "w") as f:
        f.write(tex)
    return out_tex

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LateX Error tables")
    parser.add_argument("--root", help="Root directory of the experiemnt")
    parser.add_argument("--system", help="Name of the system")
    parser.add_argument("--decimals", help="Percision of values after decimal point", default=3)

    args = parser.parse_args()

    ROOT_DIR = f"{args.root}/{args.system}"
    CSV_PATHS = sorted(glob.glob(f"{ROOT_DIR}/sparse-*.csv"))
    DECIMALS = args.decimals
    OUT_TEX = f"{'poly' if 'poly' in ROOT_DIR else 'gaussian'}-{args.system}.tex"
    SYS_MAP = {"ms": "Mass-Spring", "m2s3": "Two-Mass-Three-Spring", "hh": "Henon–Heiles", "np": "Nonlinear Pendulum"}
    KERNEL_TYPE = ''
    if ('poly' in ROOT_DIR) and (args.system != "np"):
        KERNEL_TYPE = "separable polynomial"
    elif ('poly' in ROOT_DIR) and (args.system == "np"):
        KERNEL_TYPE = "additive polynomial and Gaussian"
    else:
        KERNEL_TYPE = "Gaussian"

    CAPTION = rf"\caption{{Interpolation and extrapolation relative errors (mean $\pm$ std) for the {SYS_MAP.get(args.system)} system with a {KERNEL_TYPE} kernel for $H$}}"
    LABEL = rf"\label{{tab:{OUT_TEX.replace('.tex', '')}}}"

    generate_re_pq_table(CSV_PATHS, OUT_TEX, CAPTION, LABEL, decimals=DECIMALS)
