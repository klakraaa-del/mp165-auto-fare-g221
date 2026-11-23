import pandas as pd
import yaml


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def clean_booking(s, booking_map):
    s = s.str.strip().str.lower().str.rstrip(".")
    return s.replace(booking_map)

def fix_units(df, p):
    c = p["cleaning"]
    mask = df[c["unit_group_column"]] == c["unit_group_value"]
    df.loc[mask, c["unit_column"]] = df.loc[mask, c["unit_column"]] / c["unit_factor"]
    return df
def drop_typos(df, target, multiplier):
    limit = multiplier * df[target].median()
    return df[df[target] <= limit]
def drop_stuck(df, p):
    c = p["cleaning"]
    if not c["drop_stuck_rows"]:
        return df
    start = pd.Timestamp(c["stuck_start"])
    end = pd.Timestamp(c["stuck_end"])
    inside = df["date"].between(start, end)
    return df[~inside]


def clean_train(df, p):
    c = p["cleaning"]
    df = df.copy()
    df = fix_units(df, p)
    df["booking"] = clean_booking(df["booking"], c["booking_map"])
    df = drop_typos(df, p["features"]["target"], c["typo_multiplier"])
    df = drop_stuck(df, p)
    df = df.drop(columns=["fare_paid_receipt_rs"])
    return df.reset_index(drop=True)


def clean_features(df, p):
    df = df.copy()
    df["booking"] = clean_booking(df["booking"], p["cleaning"]["booking_map"])
    return df