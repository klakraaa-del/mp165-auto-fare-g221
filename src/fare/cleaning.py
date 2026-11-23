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
    # TODO: drop rows with date between stuck_start and stuck_end
    return df


def clean_train(df, p):
    df = df.copy()
    # TODO: call the functions above in a sensible order
    # TODO: drop the leaking column fare_paid_receipt_rs
    return df


def clean_features(df, p):
    # TODO: fix booking spelling only; must be safe on already-clean data
    return df
