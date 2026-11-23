"""Seven data checks for MP165 train.csv.
Run from the repo root:  python src/fare/data_checks.py data/raw
"""
import sys
import pandas as pd


def load(folder):
    train = pd.read_csv(f"{folder}/train.csv", parse_dates=["date"])
    hold = pd.read_csv(f"{folder}/holdout_features.csv")
    return train, hold


def typo_mask(df, target="fare_rs"):
    # a typo is a target far above the normal range (10x too big)
    return df[target] > 3 * df[target].median()


def leaking_column(df, hold, target="fare_rs"):
    """Columns missing from holdout that agree almost exactly with the target."""
    missing = [c for c in df.columns if c not in hold.columns and c != target]
    ok = ~typo_mask(df, target)  # typos would hide the agreement
    out = {}
    for c in missing:
        r = df.loc[ok, c].corr(df.loc[ok, target])
        if r > 0.99:
            out[c] = round(r, 4)
    return out


def unit_mixup(df, group="zone"):
    """Numeric columns whose mean differs hugely between group values."""
    out = {}
    for c in df.select_dtypes("number"):
        means = df.groupby(group)[c].mean()
        if means.max() / means.min() > 5:
            out[c] = (means.idxmax(), round(means.max() / means.median(), 1))
    return out


def sentinel_candidates(df):
    """Most repeated value in each numeric column; look for impossible ones."""
    return {c: df[c].value_counts().head(1).to_dict()
            for c in df.select_dtypes("number")}


def duplicate_rows(df):
    return int(df.drop(columns="record_id").duplicated().sum())


def stuck_run(df, col):
    """Longest run of identical consecutive values, ordered by date."""
    d = df.sort_values(["date", "record_id"]).reset_index(drop=True)
    run = (d[col] != d[col].shift()).cumsum()
    g = d.groupby(run).agg(val=(col, "first"), start=("date", "min"),
                           end=("date", "max"), n=("date", "size"))
    return g.sort_values("n").iloc[-1]


def count_typos(df, target="fare_rs"):
    return int(typo_mask(df, target).sum())


def clean_booking(s):
    s = s.str.strip().str.lower().str.rstrip(".")
    return s.replace({"str": "street", "sta": "stand"})


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    df, hold = load(folder)
    print("leak:", leaking_column(df, hold))
    print("unit mixup:", unit_mixup(df))
    print("sentinel candidates:", sentinel_candidates(df))
    print("duplicates:", duplicate_rows(df))
    print("stuck luggage_bags:\n", stuck_run(df, "luggage_bags"))
    print("target typos:", count_typos(df))
    print("booking levels:", clean_booking(df.booking).nunique())