"""
app.py
------
Stage 9: the Streamlit dashboard.

This file is deliberately thin. It contains no matching logic at all - it
only calls the functions in src/ and draws the results. That separation is
the point: every number shown here can be reproduced from the command line
with `python -m src.evaluation`, so the dashboard is a view of the pipeline,
not a second implementation of it.

Run:  streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluation import (
    SWEEP_THRESHOLDS,
    confusion,
    confusion_matrix_frame,
    error_examples,
    load_ground_truth,
    metrics,
    threshold_sweep,
)
from src.matching import (
    LABEL_DUPLICATE,
    LABEL_POSSIBLE,
    LABEL_UNIQUE,
    RESULT_COLUMNS,
    block_summary,
    duplicate_groups,
    exact_stage_summary,
    explain_pair,
    run_pipeline,
)
from src.preprocessing import (
    NORM_PAIRS,
    inspect,
    load_data,
    normalization_examples,
    normalization_impact,
)

st.set_page_config(page_title="Duplicate Record Detection", layout="wide")


# ----------------------------------------------------------------------
# Data loading + pipeline (cached so the sliders stay responsive)
# ----------------------------------------------------------------------

@st.cache_data
def read_csv(source):
    return load_data(source)


@st.cache_data
def run(df, weights, thresholds):
    # weights/thresholds arrive as tuples of pairs so Streamlit can hash them
    return run_pipeline(df, dict(weights), dict(thresholds))


# ----------------------------------------------------------------------
# Sidebar: the knobs
# ----------------------------------------------------------------------

st.sidebar.title("Settings")

uploaded = st.sidebar.file_uploader("Upload a CSV", type="csv")
df_raw = read_csv(uploaded if uploaded else "data/sample_data.csv")
if not uploaded:
    st.sidebar.caption("Using data/sample_data.csv")

st.sidebar.subheader("Field weights")
st.sidebar.caption(
    "A design choice, not a universal truth. Fields where two records are "
    "unlikely to agree by accident deserve more weight."
)
w_name = st.sidebar.slider("Name", 0.0, 0.5, 0.25, 0.05)
w_email = st.sidebar.slider("Email", 0.0, 0.5, 0.25, 0.05)
w_phone = st.sidebar.slider("Phone", 0.0, 0.5, 0.25, 0.05)
w_address = st.sidebar.slider("Address", 0.0, 0.5, 0.15, 0.05)
w_city = st.sidebar.slider("City", 0.0, 0.5, 0.10, 0.05)

weights = {"name": w_name, "email": w_email, "phone": w_phone,
           "address": w_address, "city": w_city}
st.sidebar.caption(
    f"Weights sum to {sum(weights.values()):.2f}. They are renormalized per "
    "pair, so this does not have to be 1.00."
)

st.sidebar.subheader("Decision thresholds")
t_dup = st.sidebar.slider("High confidence duplicate", 0.50, 1.00, 0.90, 0.01)
t_pos = st.sidebar.slider("Possible duplicate", 0.50, 1.00, 0.75, 0.01)
if t_pos > t_dup:
    st.sidebar.error("The 'possible' threshold must be below the 'duplicate' one.")
    t_pos = t_dup

thresholds = {"duplicate": t_dup, "possible": t_pos}

out = run(df_raw, tuple(sorted(weights.items())), tuple(sorted(thresholds.items())))
df_norm, results, block_stats = out["df_norm"], out["results"], out["block_stats"]


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("Duplicate Record Detection System")
st.caption(
    "Finds records that describe the same person even when they are written "
    "differently. Pipeline: inspect -> normalize -> block -> compare -> "
    "score -> classify -> evaluate."
)

tabs = st.tabs([
    "1. Overview", "2. Preprocessing", "3. Exact duplicates",
    "4. Blocking", "5. Results", "6. Compare a pair", "7. Evaluation",
])


# ----------------------------------------------------------------------
# 1. Overview
# ----------------------------------------------------------------------

with tabs[0]:
    report = inspect(df_raw)
    basic = report["basic"]
    total_missing = int(report["missing"]["missing"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", basic["n_rows"])
    c2.metric("Columns", basic["n_columns"])
    c3.metric("Missing values", total_missing)
    c4.metric("Exact duplicate rows", report["exact_duplicates"]["extra_copies"])

    st.subheader("Why inspect before processing?")
    st.markdown(
        "- **Data types** - phone and pincode are read as text, not numbers. "
        "They are identifiers that happen to be made of digits, and reading "
        "them as numbers would drop leading zeros.\n"
        "- **Missing values** - a field that is often blank makes a weak "
        "feature. We need to know that before we trust it.\n"
        "- **Unique values** - this is what tells us which fields are strong "
        "evidence, and it is where the scoring weights come from."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Missing values per column")
        st.dataframe(report["missing"], hide_index=True)
        fig = px.bar(report["missing"], x="column", y="missing",
                     title="Missing values by column")
        st.plotly_chart(fig)

    with right:
        st.subheader("Unique values per column")
        st.caption(
            "uniqueness_ratio = unique values / rows. Close to 1 means the "
            "column behaves like an identifier, so a match is strong "
            "evidence. Close to 0 means it is categorical, so a match is weak "
            "evidence because unrelated records agree by chance."
        )
        st.dataframe(report["unique"], hide_index=True)

    st.subheader("Raw data")
    st.dataframe(df_raw.head(50), hide_index=True)


# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------

with tabs[1]:
    st.subheader("Normalization: removing differences that carry no meaning")
    st.markdown(
        "Two records about the same person differ in two ways. **Meaningless** "
        "differences (capitals, spacing, +91, commas, Rd vs Road) are deleted "
        "here. **Meaningful** differences (a genuinely different phone number) "
        "are left for the similarity stage to weigh.\n\n"
        "Raw columns are never overwritten - each gets a `_norm` twin, so the "
        "dashboard can still show what was actually typed."
    )

    impact = normalization_impact(df_norm)
    st.dataframe(impact, hide_index=True)
    fig = px.bar(
        impact.melt(id_vars="column",
                    value_vars=["unique_before", "unique_after"],
                    var_name="stage", value_name="distinct_values"),
        x="column", y="distinct_values", color="stage", barmode="group",
        title="Distinct values before and after normalization",
    )
    st.plotly_chart(fig)
    st.caption(
        "City is the headline: many spellings collapse to a handful of real "
        "cities. Pincode barely moves, which honestly means the pincode rules "
        "are not doing much on this dataset."
    )

    st.subheader("What actually changed")
    column = st.selectbox("Column", [raw for raw, _ in NORM_PAIRS])
    st.dataframe(normalization_examples(df_norm, column, n=15), hide_index=True)


# ----------------------------------------------------------------------
# 3. Exact duplicates
# ----------------------------------------------------------------------

with tabs[2]:
    summary = exact_stage_summary(df_norm)
    c1, c2, c3 = st.columns(3)
    c1.metric("Exact duplicate pairs", summary["exact_pairs"],
              help="Raw rows identical byte for byte")
    c2.metric("Normalized-only pairs", summary["normalized_only_pairs"],
              help="Raw rows differ, identical after cleaning")
    c3.metric("Gain from normalization", summary["gain_from_normalization"])

    st.markdown(
        "**Exact duplicate** - the raw records are identical. Plain pandas "
        "`.duplicated()` finds these.\n\n"
        "**Normalized duplicate** - the raw records differ, but after Stage 2 "
        "they become identical. Finding these is what preprocessing bought us.\n\n"
        "Both are found by *grouping*, which uses a hash table and costs "
        "roughly O(n). Neither needs a similarity score. This is the cheap, "
        "high-precision part of the pipeline - and on its own it finds well "
        "under half of the duplicates."
    )

    groups = duplicate_groups(df_norm)
    if len(groups):
        counts_by_kind = groups["kind"].value_counts().reset_index()
        counts_by_kind.columns = ["kind", "groups"]
        fig = px.bar(counts_by_kind, x="kind", y="groups",
                     title="Duplicate groups by type")
        st.plotly_chart(fig)
        st.dataframe(groups, hide_index=True)

        st.subheader("Look at one group")
        gid = st.selectbox("Group", groups["group_id"].tolist())
        ids = [i.strip() for i in
               groups.loc[groups.group_id == gid, "record_ids"].iloc[0].split(",")]
        picked = df_norm[df_norm.record_id.isin(ids)]
        st.write("Raw:")
        st.dataframe(picked[["record_id", "name", "email", "phone",
                             "address", "city", "pincode"]], hide_index=True)
        st.write("Normalized:")
        st.dataframe(picked[["record_id", "name_norm", "email_norm", "phone_norm",
                             "address_norm", "city_norm", "pincode_norm"]],
                     hide_index=True)
    else:
        st.info("No duplicate groups found in this dataset.")


# ----------------------------------------------------------------------
# 4. Blocking
# ----------------------------------------------------------------------

with tabs[3]:
    st.subheader("Candidate generation: avoiding the n-squared trap")
    st.latex(r"\text{pairs} = \frac{n(n-1)}{2}")
    st.markdown(
        f"With **n = {block_stats['records']}** records that is "
        f"**{block_stats['all_possible_pairs']:,}** pairs. Double the data and "
        "the work goes up four times, while the number of real duplicates only "
        "doubles.\n\n"
        "Blocking files each record under a few cheap keys and only compares "
        "records that share one."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("All possible pairs", f"{block_stats['all_possible_pairs']:,}")
    c2.metric("Candidate pairs", f"{block_stats['candidate_pairs']:,}")
    c3.metric("Comparisons avoided", f"{block_stats['reduction_ratio']:.2%}")

    st.markdown(
        "**The keys used**\n"
        "1. `name token prefix + city` - one key per token, so a word swap or "
        "a typo in one token does not lose the record\n"
        "2. `normalized phone`\n"
        "3. `email user part` - dots and +tags removed\n"
        "4. `pincode + first letter of name` - a fallback for records whose "
        "phone and email are both missing\n\n"
        "A pair becomes a candidate if it shares **at least one** key, so a "
        "real duplicate is only lost if it fails every key at once."
    )
    st.dataframe(block_summary(df_norm), hide_index=True)
    st.caption(
        f"Largest block: {block_stats['largest_block']} records. Blocks bigger "
        "than the limit are dropped, because one huge block would undo the "
        "saving - a block of k records costs k(k-1)/2 pairs by itself."
    )


# ----------------------------------------------------------------------
# 5. Results
# ----------------------------------------------------------------------

with tabs[4]:
    class_counts = results["classification"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("High confidence", int(class_counts.get(LABEL_DUPLICATE, 0)))
    c2.metric("Possible", int(class_counts.get(LABEL_POSSIBLE, 0)))
    c3.metric("Probably unique", int(class_counts.get(LABEL_UNIQUE, 0)))

    st.markdown(
        "A **similarity score** and a **classification** are different things. "
        "The score is evidence - 0.83 means 'fairly similar'. The "
        "classification is an action: merge, send for review, or leave alone. "
        "Where the cut-offs go depends on what a mistake costs, not on the "
        f"data.\n\nHere: at or above **{t_dup:.2f}** we call it a duplicate; "
        f"between **{t_pos:.2f}** and **{t_dup:.2f}** a human should look."
    )

    left, right = st.columns(2)
    with left:
        bar_df = class_counts.reset_index()
        bar_df.columns = ["classification", "pairs"]
        fig = px.bar(bar_df, x="classification", y="pairs",
                     title="How the candidate pairs were classified")
        st.plotly_chart(fig)
    with right:
        fig = px.histogram(results, x="final_score", nbins=40,
                           title="Distribution of match scores")
        fig.add_vline(x=t_dup, line_dash="dash", annotation_text="duplicate")
        fig.add_vline(x=t_pos, line_dash="dot", annotation_text="possible")
        st.plotly_chart(fig)
    st.caption(
        "The score histogram is bunched at the two ends: obvious matches near "
        "1.0 and obvious non-matches low down. The thin middle is where every "
        "mistake lives, and it is what the review band exists for."
    )

    st.subheader("Detected pairs")
    choice = st.radio("Show", [LABEL_DUPLICATE, LABEL_POSSIBLE, LABEL_UNIQUE, "All"],
                      horizontal=True)
    shown = results if choice == "All" else results[results.classification == choice]
    st.dataframe(shown[RESULT_COLUMNS].head(200), hide_index=True)

    st.download_button(
        "Download duplicate_results.csv",
        data=results[RESULT_COLUMNS].to_csv(index=False).encode(),
        file_name="duplicate_results.csv",
        mime="text/csv",
    )


# ----------------------------------------------------------------------
# 6. Compare a pair
# ----------------------------------------------------------------------

with tabs[5]:
    st.subheader("Side-by-side comparison")
    flagged = results[results.classification != LABEL_UNIQUE].reset_index(drop=True)
    if not len(flagged):
        st.info("No pairs were flagged at the current thresholds.")
    else:
        labels = [
            f"{r.record_1} vs {r.record_2}  -  {r.final_score:.2f}  ({r.classification})"
            for r in flagged.itertuples()
        ]
        picked_label = st.selectbox("Pair", labels)
        row = flagged.iloc[labels.index(picked_label)]

        a = df_norm[df_norm.record_id == row.record_1].iloc[0]
        b = df_norm[df_norm.record_id == row.record_2].iloc[0]
        fields = ["name", "email", "phone", "address", "city", "pincode"]

        comparison = pd.DataFrame({
            "Field": [f.title() for f in fields],
            f"Record A ({row.record_1})": [a[f] for f in fields],
            f"Record B ({row.record_2})": [b[f] for f in fields],
            "Normalized A": [a[f + "_norm"] for f in fields],
            "Normalized B": [b[f + "_norm"] for f in fields],
        })
        st.dataframe(comparison, hide_index=True)

        st.subheader("Feature scores")
        c1, c2 = st.columns(2)
        with c1:
            feat = pd.DataFrame({
                "Feature": ["Name (fuzzy)", "Name (partial)", "Name (Jaccard)",
                            "Email (fuzzy)", "Email exact match",
                            "Phone exact match", "Address (fuzzy)",
                            "Address (Jaccard)", "City match", "Pincode match",
                            "TF-IDF cosine"],
                "Score": [row.name_similarity, row.name_partial, row.name_jaccard,
                          row.email_similarity, row.email_match, row.phone_match,
                          row.address_similarity, row.address_jaccard,
                          row.city_match, row.pincode_match, row.tfidf_cosine],
            })
            st.dataframe(feat, hide_index=True)
        with c2:
            order = ["name", "email", "phone", "address", "city"]
            comps = pd.DataFrame({
                "Component": order,
                "Score": [row[f"comp_{k}"] for k in order],
                "Weight": [weights[k] for k in order],
            })
            comps["Contribution"] = (comps["Score"] * comps["Weight"]).round(4)
            st.dataframe(comps, hide_index=True)
            st.metric("Final score", f"{row.final_score:.2f}", row.classification)
            st.caption(
                f"Compared on {int(row.fields_compared)} of 5 fields. Missing "
                "fields are dropped and the remaining weights renormalized - a "
                "blank counts as no evidence, not as evidence against."
            )

        st.subheader("Why flagged?")
        for reason in explain_pair(row):
            st.markdown(f"- {reason}")


# ----------------------------------------------------------------------
# 7. Evaluation
# ----------------------------------------------------------------------

with tabs[6]:
    st.subheader("Measuring the detector against known answers")
    try:
        true_pairs, hard_negatives = load_ground_truth()
    except FileNotFoundError:
        true_pairs = None
        st.warning("data/ground_truth.csv not found - evaluation needs labels.")

    if true_pairs:
        counts = confusion(results, true_pairs, t_dup)
        m = metrics(counts)
        total = counts["TP"] + counts["FP"] + counts["FN"] + counts["TN"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{m['precision']:.3f}",
                  help="Of the pairs we flagged, how many were really duplicates")
        c2.metric("Recall", f"{m['recall']:.3f}",
                  help="Of the real duplicates, how many did we find")
        c3.metric("F1", f"{m['f1']:.3f}")

        st.subheader("Confusion matrix")
        st.dataframe(confusion_matrix_frame(counts))
        st.markdown(
            "- **True Positive** - flagged, and really the same person\n"
            "- **False Positive** - flagged, but two different people. "
            "Usually the expensive mistake, because merging destroys data\n"
            "- **False Negative** - a real duplicate we missed\n"
            "- **True Negative** - correctly left alone\n\n"
            f"Accuracy here is {m['accuracy']:.3f}, and it is **misleading**. "
            "Flagging nothing at all would still score about "
            f"{counts['TN'] / max(total, 1):.3f}, because almost every pair "
            "really is a non-duplicate. Precision and recall ignore the True "
            "Negative count, which is exactly why we use them on imbalanced "
            "problems like this one."
        )

        st.subheader("Threshold experiment")
        sweep = threshold_sweep(results, true_pairs, SWEEP_THRESHOLDS)
        fig = px.line(
            sweep.melt(id_vars="threshold",
                       value_vars=["precision", "recall", "f1"],
                       var_name="metric", value_name="value"),
            x="threshold", y="value", color="metric", markers=True,
            title="Precision / recall / F1 against the decision threshold",
        )
        st.plotly_chart(fig)
        st.dataframe(
            sweep[["threshold", "TP", "FP", "FN", "precision", "recall", "f1"]],
            hide_index=True)
        best = sweep.loc[sweep.f1.idxmax()]
        st.markdown(
            f"Best F1 on this dataset is **{best.f1:.3f}** at threshold "
            f"**{best.threshold:.2f}**.\n\n"
            "Raising the threshold demands more evidence, so fewer pairs clear "
            "the bar: precision rises and recall falls. That is the "
            "**precision-recall trade-off**. There is no universally correct "
            "setting - it depends on whether a wrong merge or a missed "
            "duplicate costs you more."
        )

        st.subheader("Where it went wrong")
        fp_df, fn_df = error_examples(results, counts, n=10)
        st.write("**False positives** - flagged, but actually different people:")
        st.dataframe(fp_df, hide_index=True)
        st.write("**False negatives** - real duplicates we missed:")
        st.dataframe(fn_df, hide_index=True)
        st.caption(
            "These rows are more useful than the metrics. A metric tells you "
            "that the model is wrong; these tell you why, and that is what "
            "drives the next improvement."
        )
