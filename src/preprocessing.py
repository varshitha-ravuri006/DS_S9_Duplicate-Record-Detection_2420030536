"""
preprocessing.py
----------------
Stage 1: look at the data before touching it.
Stage 2: clean and normalize it.

Why inspect first?
Every decision later in the pipeline depends on what the data actually looks
like. If 20% of phone numbers are blank, a "phone match" feature is weaker
than we assumed. If `pincode` was silently read as an integer, "500001"
becomes 500001 and a leading-zero pincode like "011234" would lose its zero.
Five minutes of inspection prevents a whole class of silent bugs.

Why normalize?
Two records describing the same person are rarely written the same way. The
differences are almost always meaningless - spacing, capitals, "+91", a
comma, "Rd" instead of "Road". Normalization removes the differences that
carry no information, so that the similarity functions in Stage 5 spend their
effort on differences that actually matter.

Run:  python -m src.preprocessing
"""

import re

import pandas as pd

# Columns that describe the customer. `record_id` is deliberately excluded:
# it is a unique key we created, so including it would make every row look
# unique and exact-duplicate detection would always return zero.
FIELD_COLUMNS = ["name", "email", "phone", "address", "city", "pincode"]


def load_data(path):
    """
    Read the CSV.

    dtype=str is important. Pandas would otherwise guess that `phone` and
    `pincode` are numbers. We never do arithmetic on them - they are
    identifiers that happen to be made of digits - and treating them as
    numbers loses leading zeros and turns blanks into NaN floats.

    keep_default_na=False keeps empty cells as "" instead of NaN, which makes
    string cleaning in Stage 2 much simpler.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df


def basic_info(df):
    """Shape, column names and dtypes - the first thing to check."""
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }


def missing_report(df):
    """
    How many cells are empty per column.

    We treat "" and whitespace-only cells as missing, not just NaN. A cell
    containing "   " is missing information even though pandas calls it a
    valid string.
    """
    blank = df.apply(lambda col: col.astype(str).str.strip().eq("") | col.isna())
    counts = blank.sum()
    return pd.DataFrame({
        "column": counts.index,
        "missing": counts.values,
        "missing_pct": (counts.values / len(df) * 100).round(2),
    }).sort_values("missing", ascending=False, ignore_index=True)


def unique_report(df):
    """
    Unique values per column.

    This tells us which columns behave like identifiers and which behave like
    categories:
      - `email` with ~450 unique values out of 456 rows -> near-identifier,
        so an email match is strong evidence of a duplicate.
      - `city` with 7 unique values -> categorical, so a city match is weak
        evidence (1 in 7 unrelated records match by chance).
    This single table is what justifies the feature weights in Stage 7.
    """
    rows = []
    for col in df.columns:
        n_unique = df[col].nunique()
        rows.append({
            "column": col,
            "unique_values": n_unique,
            "uniqueness_ratio": round(n_unique / len(df), 3),
            "kind": "identifier-like" if n_unique / len(df) > 0.9 else "categorical-like",
            "sample": ", ".join(df[col].dropna().astype(str).unique()[:3]),
        })
    return pd.DataFrame(rows)


def exact_duplicate_report(df, subset=None):
    """
    Rows that are byte-for-byte identical across the customer columns.

    `duplicated()` marks every occurrence after the first, so
    `duplicated().sum()` counts the *extra* copies, not the groups.
    keep=False marks all members of a duplicate group, which is what we want
    when displaying them to the user.
    """
    subset = subset or [c for c in FIELD_COLUMNS if c in df.columns]
    extra_copies = int(df.duplicated(subset=subset).sum())
    all_members = df[df.duplicated(subset=subset, keep=False)].sort_values(subset)
    return {
        "extra_copies": extra_copies,
        "groups": int(all_members.groupby(subset, dropna=False).ngroups) if len(all_members) else 0,
        "rows": all_members,
    }


def inspect(df):
    """Everything above in one call - used by the Streamlit dashboard."""
    return {
        "basic": basic_info(df),
        "missing": missing_report(df),
        "unique": unique_report(df),
        "exact_duplicates": exact_duplicate_report(df),
    }


# ======================================================================
# STAGE 2 - NORMALIZATION
# ======================================================================
#
# Rule we follow everywhere below: NEVER overwrite a raw column.
# Each raw column gets a `_norm` twin. Reasons:
#   1. The dashboard must show the user what they actually typed.
#   2. If a normalization rule turns out to be wrong, we can fix it and
#      re-run without having destroyed the source data.
#   3. In data engineering terms this is the difference between the raw
#      layer and the cleaned layer of a pipeline.

# Words that appear in a name but carry no identity information.
NAME_TITLES = {"mr", "mrs", "ms", "miss", "dr", "shri", "smt", "prof", "sri"}

# Short forms seen in addresses -> the one spelling we standardise on.
# We expand short to long (not long to short) so the result stays readable.
ADDRESS_SYNONYMS = {
    "rd": "road", "st": "street", "str": "street", "ngr": "nagar",
    "cly": "colony", "gdn": "garden", "sec": "sector", "apt": "apartment",
    "flr": "floor", "blk": "block", "opp": "opposite", "nr": "near",
    "hno": "house", "ph": "phase", "ext": "extension",
}

# A controlled vocabulary for cities: every spelling we have seen mapped to
# one canonical name. This is hand-built, which is a real limitation - a new
# spelling ("Hydrabad") would not be recognised until we add it.
CITY_SYNONYMS = {
    "hyderabad": "hyderabad", "hyd": "hyderabad", "hydrabad": "hyderabad",
    "secunderabad": "hyderabad",
    "bengaluru": "bengaluru", "bangalore": "bengaluru", "blr": "bengaluru",
    "banglore": "bengaluru", "bengalore": "bengaluru",
    "mumbai": "mumbai", "bombay": "mumbai", "mum": "mumbai", "bom": "mumbai",
    "chennai": "chennai", "madras": "chennai", "maa": "chennai",
    "pune": "pune", "poona": "pune", "pnq": "pune",
    "delhi": "delhi", "new delhi": "delhi", "del": "delhi", "ncr": "delhi",
    "kolkata": "kolkata", "calcutta": "kolkata", "ccu": "kolkata",
}

# Email providers that are the same mailbox under two names.
DOMAIN_SYNONYMS = {"googlemail.com": "gmail.com"}


def _squash(text):
    """Collapse any run of whitespace into a single space and trim."""
    return re.sub(r"\s+", " ", text).strip()


def _join_initials(text):
    """
    Glue dotted initials back together: "m.g. road" -> "mg road".

    This must run BEFORE we turn punctuation into spaces. Otherwise
    "M.G. Road" becomes "m g road" while "MG Rd" becomes "mg road", and the
    two never match - which is exactly the bug this project exists to catch.

    The pattern only fires on a SINGLE letter followed by a dot, so "Dr."
    and "No." are left alone.
    """
    return re.sub(r"\b([a-z])\.", r"\1", text)


def normalize_name(value):
    """
    "  Mr. RAVI   Kumar " -> "ravi kumar"

    Steps: lowercase -> punctuation becomes a space -> drop title words
    -> collapse spaces.

    Note what we do NOT do: we do not reorder the words. "Kumar Ravi" stays
    "kumar ravi" and will not equal "ravi kumar". That is handled later by
    RapidFuzz's token_sort_ratio, because reordering here would destroy
    information the similarity function can use more carefully.
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    text = _join_initials(value.lower())              # "a.b. kumar" -> "ab kumar"
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in NAME_TITLES]
    return _squash(" ".join(tokens))


def normalize_email(value):
    """
    "  RAVI@GMAIL.COM " -> "ravi@gmail.com"

    Only lowercasing and trimming - the safe, universally-true part.
    The aggressive part (dropping dots and +tags) lives in
    `email_user_part` below, kept separate so it is a visible choice.
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    return value.strip().lower()


def email_user_part(normalized_email):
    """
    "ravi.kumar+shop@gmail.com" -> "ravikumar"

    Gmail ignores dots and treats everything after '+' as a tag, so
    ravi.kumar@gmail.com and ravikumar@gmail.com are the same mailbox.
    This is a provider-specific assumption, which is exactly why it is a
    separate function instead of being buried inside normalize_email.
    """
    if "@" not in normalized_email:
        return ""
    local = normalized_email.split("@", 1)[0]
    local = local.split("+", 1)[0]                     # drop the +tag
    return re.sub(r"[^a-z0-9]", "", local)             # drop . _ -


def email_domain_part(normalized_email):
    """"ravi@googlemail.com" -> "gmail.com". Also used for blocking in Stage 4."""
    if "@" not in normalized_email:
        return ""
    domain = normalized_email.split("@", 1)[1]
    return DOMAIN_SYNONYMS.get(domain, domain)


def normalize_phone(value):
    """
    "+91-98765-43210" -> "9876543210"

    Keep digits only, then reduce to the 10-digit national number:
      12 digits starting 91  -> country code, drop it
      11 digits starting 0   -> trunk prefix, drop it
    Anything that is not 10 digits afterwards is returned as-is; it is
    probably a landline or a typo and we would rather see it than silently
    invent a number.
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits


def normalize_address(value):
    """
    "12, M.G. Road, Hyderabad" -> "12 mg road hyderabad"
    "12 MG Rd Hyderabad"       -> "12 mg road hyderabad"

    Lowercase -> punctuation to space -> expand each abbreviation token.
    We replace whole tokens only. A blind str.replace("st", "street") would
    turn "station" into "streetation".
    """
    if not isinstance(value, str) or not value.strip():
        return ""
    text = _join_initials(value.lower())              # "m.g. road" -> "mg road"
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [ADDRESS_SYNONYMS.get(t, t) for t in text.split()]
    return _squash(" ".join(tokens))


def normalize_city(value):
    """"HYD" -> "hyderabad". Falls back to the cleaned text if unknown."""
    if not isinstance(value, str) or not value.strip():
        return ""
    text = _squash(re.sub(r"[^a-z0-9\s]", " ", value.lower()))
    return CITY_SYNONYMS.get(text, text)


def normalize_pincode(value):
    """Digits only. Indian pincodes are 6 digits; we keep whatever remains."""
    if not isinstance(value, str) or not value.strip():
        return ""
    return re.sub(r"\D", "", value)


def add_normalized_columns(df):
    """
    Return a copy of df with the cleaned columns added alongside the raw ones.

    Added columns:
        name_norm, email_norm, email_user, email_domain,
        phone_norm, address_norm, city_norm, pincode_norm
    """
    out = df.copy()
    out["name_norm"] = out["name"].map(normalize_name)
    out["email_norm"] = out["email"].map(normalize_email)
    out["email_user"] = out["email_norm"].map(email_user_part)
    out["email_domain"] = out["email_norm"].map(email_domain_part)
    out["phone_norm"] = out["phone"].map(normalize_phone)
    out["address_norm"] = out["address"].map(normalize_address)
    out["city_norm"] = out["city"].map(normalize_city)
    out["pincode_norm"] = out["pincode"].map(normalize_pincode)
    return out


NORM_PAIRS = [
    ("name", "name_norm"),
    ("email", "email_norm"),
    ("phone", "phone_norm"),
    ("address", "address_norm"),
    ("city", "city_norm"),
    ("pincode", "pincode_norm"),
]


def normalization_impact(df_norm):
    """
    How many distinct spellings collapsed into how many distinct values.

    This is the evidence that normalization did something. If a column's
    unique count barely drops, the rules for that column are not earning
    their place.
    """
    rows = []
    for raw, norm in NORM_PAIRS:
        before, after = df_norm[raw].nunique(), df_norm[norm].nunique()
        rows.append({
            "column": raw,
            "unique_before": before,
            "unique_after": after,
            "collapsed": before - after,
            "reduction_pct": round((before - after) / before * 100, 1) if before else 0.0,
        })
    return pd.DataFrame(rows)


def normalization_examples(df_norm, column, n=5):
    """
    Rows where normalization changed the value - for the dashboard.

    Most rows change only by lowercasing, which is not interesting to look
    at. So we show "structural" changes first: rows where lowercasing and
    trimming alone would NOT have produced the normalized value, i.e. where
    a rule about punctuation, abbreviations or country codes actually fired.
    """
    raw, norm = column, dict(NORM_PAIRS)[column]
    simple = df_norm[raw].str.strip().str.lower()
    structural = df_norm[simple != df_norm[norm]]
    cosmetic = df_norm[(simple == df_norm[norm]) & (df_norm[raw] != df_norm[norm])]
    picked = pd.concat([structural, cosmetic]).head(n)
    return picked[["record_id", raw, norm]]


if __name__ == "__main__":
    pd.set_option("display.width", 140)

    df = load_data("data/sample_data.csv")
    report = inspect(df)

    print("--- basic info ---")
    for k, v in report["basic"].items():
        print(f"  {k:<10}: {v}")

    print("\n--- missing values ---")
    print(report["missing"].to_string(index=False))

    print("\n--- unique values ---")
    print(report["unique"].to_string(index=False))

    print("\n--- exact duplicates ---")
    ed = report["exact_duplicates"]
    print(f"  extra copies: {ed['extra_copies']}  (in {ed['groups']} groups)")
    print(ed["rows"].head(6).to_string(index=False))

    # ---------------- Stage 2 ----------------
    df_norm = add_normalized_columns(df)

    print("\n--- normalization impact ---")
    print(normalization_impact(df_norm).to_string(index=False))

    for col in ["name", "phone", "address", "city"]:
        print(f"\n--- what changed: {col} ---")
        print(normalization_examples(df_norm, col, n=4).to_string(index=False))
