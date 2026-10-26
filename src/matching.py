"""
matching.py
-----------
Stage 3: exact duplicate detection.
Stage 4: candidate generation (blocking).
Stage 7: feature engineering, weighted scoring, classification.

There are two kinds of duplicate we can find WITHOUT any similarity score:

  Exact duplicate
      The raw rows are byte-for-byte identical.
      Pandas alone finds these: df.duplicated().

  Normalized duplicate
      The raw rows are different, but after Stage 2 cleaning the records
      become identical. Example from our dataset:

          R0207  SNEHA MENON  (+91) 6500978820  246 Banjara St  Mumbai
          R0365  sneha menon        6500978820  246 Banjara St  MUM
                          -> both become
                 sneha menon  6500978820  246 banjara street andheri  mumbai

Both are found by GROUPING, not by comparing. Grouping uses a hash table, so
it costs roughly O(n). Comparing every pair costs O(n^2). We do the cheap
exact work first and leave only the genuinely hard cases for the expensive
fuzzy matching in Stage 5.

Run:  python -m src.matching
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from src.preprocessing import (
    FIELD_COLUMNS,
    add_normalized_columns,
    load_data,
)
from src.similarity import (
    build_profile_text,
    build_tfidf_matrix,
    exact_match,
    fuzzy_similarity,
    jaccard_similarity,
    pairwise_cosine,
    partial_fuzzy_similarity,
)

# The cleaned twins of FIELD_COLUMNS, used as the grouping key.
NORM_COLUMNS = [
    "name_norm", "email_norm", "phone_norm",
    "address_norm", "city_norm", "pincode_norm",
]


def _pairs_from_groups(df, subset):
    """
    Every within-group pair of record_ids, for groups of size > 1.

    A group of 3 identical records produces 3 pairs (AB, AC, BC), because
    duplicate detection is a statement about pairs of records, and the
    ground truth is stored as pairs too.
    """
    pairs = []
    for _, group in df.groupby(subset, dropna=False, sort=False):
        if len(group) > 1:
            ids = sorted(group["record_id"])
            pairs.extend(combinations(ids, 2))
    return pairs


def exact_duplicate_pairs(df):
    """Pairs whose RAW field values are identical."""
    return _pairs_from_groups(df, FIELD_COLUMNS)


def normalized_duplicate_pairs(df_norm):
    """
    Pairs whose NORMALIZED field values are identical.

    Note this is a superset of the exact pairs: if two raw rows are
    identical, their normalized rows must be identical too. So we never
    lose an exact duplicate by normalizing.
    """
    return _pairs_from_groups(df_norm, NORM_COLUMNS)


def deterministic_duplicates(df_norm):
    """
    All pairs findable without any similarity score, labelled by how.

    Returns a DataFrame: record_1, record_2, detected_by
    where detected_by is "exact" or "normalized".
    """
    exact = set(exact_duplicate_pairs(df_norm))
    normalized = set(normalized_duplicate_pairs(df_norm))

    rows = [
        {"record_1": a, "record_2": b,
         "detected_by": "exact" if (a, b) in exact else "normalized"}
        for a, b in sorted(normalized)
    ]
    return pd.DataFrame(rows, columns=["record_1", "record_2", "detected_by"])


def duplicate_groups(df_norm):
    """
    A group-level view for the dashboard: one row per cluster of identical
    records, showing whether cleaning was needed to find it.

    A group is labelled "exact" only if EVERY member is byte-identical.
    A group with two identical rows plus one differently-spelled row is
    labelled "normalized", because normalization is what pulled it together.
    """
    rows = []
    grouped = df_norm.groupby(NORM_COLUMNS, dropna=False, sort=False)
    for group_id, (_, group) in enumerate(grouped, start=1):
        if len(group) < 2:
            continue
        raw_identical = group[FIELD_COLUMNS].drop_duplicates().shape[0] == 1
        rows.append({
            "group_id": group_id,
            "size": len(group),
            "kind": "exact" if raw_identical else "normalized",
            "record_ids": ", ".join(sorted(group["record_id"])),
            "name": group["name_norm"].iloc[0],
            "phone": group["phone_norm"].iloc[0],
            "city": group["city_norm"].iloc[0],
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["kind", "size"], ascending=[True, False], ignore_index=True) \
        if len(out) else out


def exact_stage_summary(df_norm):
    """Headline numbers for the dashboard."""
    pairs = deterministic_duplicates(df_norm)
    groups = duplicate_groups(df_norm)
    n_exact = int((pairs["detected_by"] == "exact").sum())
    n_norm = int((pairs["detected_by"] == "normalized").sum())
    return {
        "exact_pairs": n_exact,
        "normalized_only_pairs": n_norm,
        "total_deterministic_pairs": len(pairs),
        "exact_groups": int((groups["kind"] == "exact").sum()) if len(groups) else 0,
        "normalized_groups": int((groups["kind"] == "normalized").sum()) if len(groups) else 0,
        "gain_from_normalization": f"{n_norm / n_exact:.1f}x" if n_exact else "n/a",
    }


# ======================================================================
# STAGE 4 - CANDIDATE GENERATION (BLOCKING)
# ======================================================================
#
# The problem:
#     456 records -> 456 * 455 / 2 = 103,740 pairs to compare.
#     Double the data to 912 records and it becomes 415,416 pairs.
#     The work grows with n^2, but the number of real duplicates only
#     grows with n. We would be spending almost all of that effort
#     comparing records that obviously have nothing to do with each other.
#
# The idea (blocking):
#     Only compare records that already agree on something cheap.
#     "Ravi Kumar, Hyderabad" and "Meera Nair, Kolkata" share no name
#     token, no phone, no email - we can rule them out without computing
#     a single similarity score.
#
# The trade-off:
#     Blocking can LOSE a true duplicate if the pair agrees on nothing we
#     block on. So we use several independent keys and take the union: a
#     pair becomes a candidate if it shares AT LEAST ONE key. A duplicate
#     is only lost if it fails every key at once.

# Blocks bigger than this are dropped. A key shared by 80 records would
# alone generate 3,160 pairs and is clearly not selective enough to be
# useful. Real systems do the same thing.
MAX_BLOCK_SIZE = 50

# Shorter prefixes make bigger, less selective blocks; longer prefixes are
# more selective but break on typos in the first few letters.
NAME_PREFIX = 3


KEY_TYPES = ["name_city", "phone", "email", "pin_name"]


def blocking_keys(row, key_types=None):
    """
    The cheap keys one record is filed under. A record goes into several
    blocks, not one - that is what makes the scheme tolerant of errors.

    Key 1  name token prefix + city
           One key PER TOKEN, so "ravi kumar" is filed under both "rav" and
           "kum". This survives word swaps ("Kumar Ravi" shares both keys)
           and a typo in one token ("Arvi Kumar" still shares "kum").

    Key 2  the normalized phone number
           Stage 2 already reduced "+91-98765-43210" and "09876543210" to
           the same 10 digits, so an exact key works here.

    Key 3  the email user part
           "ravi.kumar+shop@gmail.com" and "ravikumar@googlemail.com" both
           reduce to "ravikumar", so again an exact key is enough.

    Key 4  pincode + first letter of the name
           A fallback for records whose phone and email are missing.
           Pincode alone would be too broad (51 values for 456 records).

    Keys 2 and 3 do not involve the city, deliberately: they are the safety
    net for a person who moved to a different city.

    `key_types` restricts which keys are used - only needed for the
    "what if we used just one key?" experiment at the bottom of this file.
    """
    use = set(key_types or KEY_TYPES)
    keys = set()

    if "name_city" in use:
        for token in row.name_norm.split():
            if len(token) >= NAME_PREFIX and row.city_norm:
                keys.add(f"name_city:{token[:NAME_PREFIX]}|{row.city_norm}")

    if "phone" in use and row.phone_norm:
        keys.add(f"phone:{row.phone_norm}")

    if "email" in use and row.email_user:
        keys.add(f"email:{row.email_user}")

    if "pin_name" in use and row.pincode_norm and row.name_norm:
        keys.add(f"pin_name:{row.pincode_norm}|{row.name_norm[0]}")

    return keys


def build_blocks(df_norm, key_types=None):
    """key -> list of record_ids filed under it."""
    index = defaultdict(list)
    for row in df_norm.itertuples():
        for key in blocking_keys(row, key_types):
            index[key].append(row.record_id)
    return index


def generate_candidate_pairs(df_norm, max_block_size=MAX_BLOCK_SIZE, key_types=None):
    """
    The union of all within-block pairs.

    Returns (pairs, stats) where pairs is a sorted list of (id_1, id_2).
    Using a set means a pair found under three different keys is still
    compared only once.
    """
    index = build_blocks(df_norm, key_types)
    pairs = set()
    oversized = []

    for key, ids in index.items():
        if len(ids) < 2:
            continue                      # a block of one produces no pair
        if len(ids) > max_block_size:
            oversized.append((key, len(ids)))
            continue
        pairs.update(combinations(sorted(ids), 2))

    n = len(df_norm)
    all_pairs = n * (n - 1) // 2
    stats = {
        "records": n,
        "all_possible_pairs": all_pairs,
        "candidate_pairs": len(pairs),
        "comparisons_avoided": all_pairs - len(pairs),
        "reduction_ratio": round(1 - len(pairs) / all_pairs, 4),
        "blocks_used": sum(1 for ids in index.values() if 2 <= len(ids) <= max_block_size),
        "blocks_skipped_too_large": len(oversized),
        "largest_block": max((len(v) for v in index.values()), default=0),
    }
    return sorted(pairs), stats


def block_summary(df_norm, max_block_size=MAX_BLOCK_SIZE):
    """Per key-type breakdown, so the dashboard can show which key does the work."""
    index = build_blocks(df_norm)
    rows = defaultdict(lambda: {"blocks": 0, "pairs": 0, "largest": 0})
    for key, ids in index.items():
        kind = key.split(":", 1)[0]
        if 2 <= len(ids) <= max_block_size:
            rows[kind]["blocks"] += 1
            rows[kind]["pairs"] += len(ids) * (len(ids) - 1) // 2
            rows[kind]["largest"] = max(rows[kind]["largest"], len(ids))
    return pd.DataFrame(
        [{"key_type": k, **v} for k, v in rows.items()]
    ).sort_values("pairs", ascending=False, ignore_index=True)


def blocking_recall(pairs, true_pairs):
    """
    Pair completeness: the fraction of real duplicates that survived blocking.

    This is the number that matters. Blocking is only allowed to throw away
    work, never answers - any true pair missing here can NEVER be found by
    the later stages, no matter how good the similarity function is.
    """
    kept = set(pairs) & set(true_pairs)
    return {
        "true_pairs": len(true_pairs),
        "kept_by_blocking": len(kept),
        "lost_by_blocking": len(true_pairs) - len(kept),
        "pair_completeness": round(len(kept) / len(true_pairs), 4) if true_pairs else 0.0,
    }


# ======================================================================
# STAGE 7 - FEATURE ENGINEERING
# ======================================================================
#
# A candidate pair is two rows of raw text. A scoring rule needs NUMBERS.
# Feature engineering is the step that turns one into the other: for each
# pair we produce a small fixed-length vector of comparison scores.
#
# Why each feature is worth having, and how much to trust it. The
# justification comes from the uniqueness report back in Stage 1:
#
#   phone_match       400 distinct values / 456 records.  STRONG.
#                     Two records agreeing by accident is very unlikely.
#   email_match       385 distinct.                        STRONG.
#   name_similarity   375 distinct, but names genuinely repeat between
#                     different people and typos are common.  MEDIUM.
#   address_similarity                                      MEDIUM.
#                     Family members share an address, and "12 MG Road"
#                     vs "98 MG Road" scores 0.89 while being a different
#                     house.
#   city_match        only 7 distinct values.               WEAK.
#                     Two unrelated records match 1 time in 7 by chance.
#   pincode_match     51 distinct.                          WEAK-MEDIUM.
#
# The rule of thumb: a feature is strong when accidental agreement is
# unlikely, which is another way of saying the column has high cardinality.

FEATURE_COLUMNS = [
    "name_similarity", "name_partial", "name_jaccard",
    "email_similarity", "email_match",
    "phone_match",
    "address_similarity", "address_jaccard",
    "city_match", "pincode_match",
    "tfidf_cosine",
]


def build_features(df_norm, pairs):
    """
    One row of numeric features per candidate pair.

    Implementation note: we look the records up ONCE into plain lists and
    index by position. Calling df.loc[record_id] inside the loop would be
    far slower, and at 1,195 pairs x 11 features that difference is real.
    """
    pos = {rid: i for i, rid in enumerate(df_norm["record_id"])}
    col = {c: df_norm[c].tolist() for c in df_norm.columns}

    rows_a = [pos[a] for a, _ in pairs]
    rows_b = [pos[b] for _, b in pairs]

    # TF-IDF is built once over the whole dataset, then read per pair.
    # The IDF weights must come from ALL records - that is what makes a
    # word "rare" or "common" in the first place.
    _, tfidf = build_tfidf_matrix(build_profile_text(df_norm))
    cosines = pairwise_cosine(tfidf, rows_a, rows_b)

    records = []
    for k, (ia, ib) in enumerate(zip(rows_a, rows_b)):
        name_a, name_b = col["name_norm"][ia], col["name_norm"][ib]
        addr_a, addr_b = col["address_norm"][ia], col["address_norm"][ib]

        records.append({
            "record_1": col["record_id"][ia],
            "record_2": col["record_id"][ib],

            # --- name: three views of the same field ---------------
            "name_similarity": fuzzy_similarity(name_a, name_b),
            "name_partial": partial_fuzzy_similarity(name_a, name_b),
            "name_jaccard": jaccard_similarity(name_a, name_b),

            # --- email: fuzzy score AND a hard match flag -----------
            # email_match uses the canonical user part, so
            # ravi.kumar@gmail.com == ravikumar@googlemail.com
            "email_similarity": fuzzy_similarity(
                col["email_norm"][ia], col["email_norm"][ib]),
            "email_match": exact_match(
                col["email_user"][ia] + "@" + col["email_domain"][ia]
                if col["email_user"][ia] else "",
                col["email_user"][ib] + "@" + col["email_domain"][ib]
                if col["email_user"][ib] else ""),

            # --- phone: exact only ---------------------------------
            # A fuzzy phone score would be actively harmful:
            # 9876543210 and 9876543211 are 90% similar as strings and
            # belong to two different people.
            "phone_match": exact_match(
                col["phone_norm"][ia], col["phone_norm"][ib]),

            # --- address: character view + token view --------------
            "address_similarity": fuzzy_similarity(addr_a, addr_b),
            "address_jaccard": jaccard_similarity(addr_a, addr_b),

            "city_match": exact_match(
                col["city_norm"][ia], col["city_norm"][ib]),
            "pincode_match": exact_match(
                col["pincode_norm"][ia], col["pincode_norm"][ib]),

            "tfidf_cosine": float(cosines[k]),
        })

    return pd.DataFrame(records)


# ======================================================================
# STAGE 13 / 14 - MATCH SCORE AND CLASSIFICATION
# ======================================================================
#
# These weights are a DESIGN CHOICE, not a law of nature. They encode the
# judgement above: identifier-like fields get more weight than categorical
# ones. They are deliberately easy to change - the dashboard exposes them
# as sliders, and a different dataset would want different numbers.
#
# A more advanced project would LEARN these weights with logistic
# regression on the ground truth instead of setting them by hand. We do not,
# because a hand-set weight is something we can explain and defend, and
# because fitting weights on a 456-row dataset we generated ourselves would
# mostly be learning our own generator.

DEFAULT_WEIGHTS = {
    "name": 0.25,
    "email": 0.25,
    "phone": 0.25,
    "address": 0.15,
    "city": 0.10,
}

DEFAULT_THRESHOLDS = {"duplicate": 0.90, "possible": 0.75}

LABEL_DUPLICATE = "High Confidence Duplicate"
LABEL_POSSIBLE = "Possible Duplicate"
LABEL_UNIQUE = "Probably Unique"


def _component_scores(features):
    """
    Collapse the 11 raw features into the 5 components the weights refer to.

    name    : best of the fuzzy and the partial (abbreviation) view
    email   : an exact match on the canonical address is worth a full 1.0,
              otherwise fall back to the fuzzy score
    phone   : exact only
    address : half character-similarity, half token-similarity, because
              Stage 5 showed each catches what the other misses
    city    : exact on the canonical city name
    """
    comp = pd.DataFrame(index=features.index)
    comp["name"] = features[["name_similarity", "name_partial"]].max(axis=1)
    comp["email"] = features["email_match"].where(
        features["email_match"] == 1.0, features["email_similarity"])
    comp["phone"] = features["phone_match"]
    comp["address"] = features[["address_similarity", "address_jaccard"]].mean(axis=1)
    comp["city"] = features["city_match"]
    return comp


def score_pairs(features, weights=None):
    """
    Weighted average of the available components.

    The important part is how MISSING values are handled. If a record has no
    phone number, `phone` is NaN. We do not score it 0 - that would punish
    the pair for information nobody ever had. Instead we drop that component
    and RENORMALIZE over the weights that remain:

        score = sum(w_i * f_i) / sum(w_i)     over available i only

    Example: a pair with no email on either side is scored on
    name/phone/address/city, whose weights sum to 0.75, and we divide by
    0.75 rather than 1.00. So a perfect match on everything else still
    scores 1.00, not 0.75.

    The honest downside: with fewer fields available the score rests on less
    evidence, but the number does not say so. The dashboard therefore also
    shows how many fields each pair was actually compared on.
    """
    weights = weights or DEFAULT_WEIGHTS
    comp = _component_scores(features)

    w = pd.Series(weights)
    available = comp.notna()
    weight_matrix = available.mul(w, axis=1)
    total_weight = weight_matrix.sum(axis=1)

    weighted_sum = (comp.fillna(0) * weight_matrix).sum(axis=1)
    score = np.where(total_weight > 0, weighted_sum / total_weight.replace(0, np.nan), 0.0)

    out = features.copy()
    for c in comp.columns:
        out[f"comp_{c}"] = comp[c]
    out["fields_compared"] = available.sum(axis=1)
    out["final_score"] = np.round(score, 4)
    return out


def classify(scored, thresholds=None):
    """
    Turn a continuous score into a decision.

    A similarity score and a classification are different things. The score
    is evidence: 0.83 means "fairly similar". The classification is an
    ACTION: merge, send for human review, or leave alone. Turning 0.83 into
    "Possible Duplicate" requires choosing where the cut-offs go, and that
    choice depends on what a mistake costs - not on the data.

    Merging two customers wrongly is usually worse than missing a duplicate,
    so the upper threshold is set high (0.90) and everything between 0.75
    and 0.90 goes to a human instead of being decided automatically.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    out = scored.copy()
    out["classification"] = np.select(
        [out["final_score"] >= t["duplicate"], out["final_score"] >= t["possible"]],
        [LABEL_DUPLICATE, LABEL_POSSIBLE],
        default=LABEL_UNIQUE,
    )
    return out


def explain_pair(row):
    """
    Plain-language reasons for one pair - the "Why flagged?" panel.

    Written by reading the component scores back out, so the explanation can
    never disagree with the score that produced it.
    """
    reasons = []
    if row.get("phone_match") == 1.0:
        reasons.append("Phone numbers are identical after normalization (strong evidence)")
    elif row.get("phone_match") == 0.0:
        reasons.append("Phone numbers are different (evidence AGAINST a match)")
    else:
        reasons.append("Phone missing on at least one record - no evidence either way")

    if row.get("email_match") == 1.0:
        reasons.append("Email addresses are the same mailbox (strong evidence)")
    elif pd.notna(row.get("email_similarity")) and row["email_similarity"] >= 0.85:
        reasons.append(f"Email addresses are {row['email_similarity']:.0%} similar")
    elif pd.notna(row.get("email_similarity")):
        reasons.append(f"Email addresses differ ({row['email_similarity']:.0%} similar)")

    name = row.get("comp_name")
    if pd.notna(name):
        word = "highly similar" if name >= 0.85 else "moderately similar" if name >= 0.7 else "different"
        reasons.append(f"Names are {word} ({name:.0%})")

    addr = row.get("comp_address")
    if pd.notna(addr):
        word = "nearly the same" if addr >= 0.85 else "partly similar" if addr >= 0.6 else "different"
        reasons.append(f"Addresses are {word} ({addr:.0%})")

    if row.get("city_match") == 1.0:
        reasons.append("Same city, but city is weak evidence (only 7 possible values)")

    if row.get("fields_compared", 5) < 5:
        reasons.append(
            f"Only {int(row['fields_compared'])} of 5 fields could be compared - "
            "the score rests on less evidence than usual")
    return reasons


def run_pipeline(df_raw, weights=None, thresholds=None, max_block_size=MAX_BLOCK_SIZE):
    """
    The whole thing end to end. Used by app.py and by evaluation.py.

    CSV -> normalize -> blocking -> features -> score -> classify
    """
    df_norm = add_normalized_columns(df_raw)
    pairs, block_stats = generate_candidate_pairs(df_norm, max_block_size)
    features = build_features(df_norm, pairs)
    scored = classify(score_pairs(features, weights), thresholds)
    scored = scored.sort_values("final_score", ascending=False, ignore_index=True)
    return {
        "df_norm": df_norm,
        "pairs": pairs,
        "block_stats": block_stats,
        "results": scored,
    }


# Column order for the downloadable duplicate_results.csv
RESULT_COLUMNS = [
    "record_1", "record_2",
    "name_similarity", "email_similarity", "phone_match",
    "address_similarity", "city_match",
    "final_score", "classification",
]


if __name__ == "__main__":
    pd.set_option("display.width", 160)

    df = add_normalized_columns(load_data("data/sample_data.csv"))
    pairs = deterministic_duplicates(df)

    print("--- stage 3 summary ---")
    for k, v in exact_stage_summary(df).items():
        print(f"  {k:<28}: {v}")

    print("\n--- duplicate groups ---")
    print(duplicate_groups(df).head(8).to_string(index=False))

    # How far does exact matching alone get us? Compare against ground truth.
    truth = pd.read_csv("data/ground_truth.csv")
    true_pairs = {
        (r.record_id_1, r.record_id_2)
        for r in truth[truth.is_duplicate == 1].itertuples()
    }
    found = {(r.record_1, r.record_2) for r in pairs.itertuples()}

    print("\n--- how much of the problem is solved so far? ---")
    print(f"  true duplicate pairs in ground truth : {len(true_pairs)}")
    print(f"  found by exact + normalized matching : {len(found & true_pairs)}")
    print(f"  wrongly flagged (false positives)    : {len(found - true_pairs)}")
    print(f"  still missing (need fuzzy matching)  : {len(true_pairs - found)}")
    print(f"  recall so far                        : {len(found & true_pairs) / len(true_pairs):.1%}")

    # ---------------- Stage 4 ----------------
    candidates, stats = generate_candidate_pairs(df)

    print("\n--- stage 4: blocking ---")
    for k, v in stats.items():
        print(f"  {k:<26}: {v:,}" if isinstance(v, int) else f"  {k:<26}: {v}")

    print("\n--- which key does the work? ---")
    print(block_summary(df).to_string(index=False))

    print("\n--- did blocking lose any real duplicates? ---")
    for k, v in blocking_recall(candidates, true_pairs).items():
        print(f"  {k:<20}: {v}")

    lost = sorted(true_pairs - set(candidates))
    if lost:
        print("\n  examples of lost pairs:")
        for a, b in lost[:3]:
            print(df[df.record_id.isin([a, b])][
                ["record_id", "name_norm", "email_user", "phone_norm", "city_norm"]
            ].to_string(index=False), "\n")

    # The trade-off, measured: a narrower blocking scheme does less work but
    # silently throws away real duplicates it can never get back.
    print("\n--- what if we blocked on only ONE key? ---")
    experiments = [["name_city"], ["phone"], ["email"], ["pin_name"], KEY_TYPES]
    rows = []
    for keys in experiments:
        cand, st = generate_candidate_pairs(df, key_types=keys)
        rec = blocking_recall(cand, true_pairs)
        rows.append({
            "keys_used": "+".join(keys) if len(keys) < 4 else "ALL FOUR",
            "candidate_pairs": st["candidate_pairs"],
            "reduction": f"{st['reduction_ratio']:.2%}",
            "true_pairs_kept": rec["kept_by_blocking"],
            "pair_completeness": f"{rec['pair_completeness']:.1%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------- Stage 7 ----------------
    out = run_pipeline(load_data("data/sample_data.csv"))
    results = out["results"]

    print("\n--- stage 7: classification counts ---")
    print(results["classification"].value_counts().to_string())

    print("\n--- top scoring pairs ---")
    print(results.head(5)[
        ["record_1", "record_2", "comp_name", "comp_email", "comp_phone",
         "comp_address", "final_score", "classification"]
    ].to_string(index=False))

    print("\n--- borderline pairs (this is where the errors live) ---")
    band = results[results["classification"] == LABEL_POSSIBLE]
    print(band.head(5)[
        ["record_1", "record_2", "comp_name", "comp_email", "comp_phone",
         "comp_address", "final_score"]
    ].to_string(index=False))

    print("\n--- worked example: why was the top pair flagged? ---")
    top = results.iloc[0]
    print(f"  {top['record_1']} vs {top['record_2']}  score {top['final_score']:.2f}")
    for reason in explain_pair(top):
        print(f"   - {reason}")
