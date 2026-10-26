"""
similarity.py
-------------
Stage 5 and 6: how similar are two field values?

Stage 3 could only answer "identical or not". That is a yes/no question and
it found only 30% of the duplicates. Everything below answers a softer
question - "how close are these two strings?" - and returns a number
between 0 and 1.

Three different methods, because they fail in different ways:

  A. Fuzzy string similarity (RapidFuzz)
        character-level. Good at typos.        "ravi kumar" vs "arvi kumar"

  B. Jaccard similarity
        token-level, order-blind. Good at missing/extra words.
        "12 mg road kondapur" vs "12 mg road"

  C. TF-IDF + cosine similarity (scikit-learn)
        token-level but WEIGHTED, so rare words count more than common
        ones. Good at ignoring filler like "road" and "nagar".

Missing values are returned as NaN, never as 0. Two blank phone numbers do
not agree - we simply have no evidence. Stage 7 handles NaN by redistributing
that field's weight, which is very different from scoring it zero.

Run:  python -m src.similarity
"""

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------------------
# A. Fuzzy string similarity
# ----------------------------------------------------------------------


def fuzzy_similarity(a, b):
    """
    The better of two RapidFuzz views, rescaled from 0-100 to 0-1.

    What the underlying measure does: it counts the minimum number of
    single-character edits (insert / delete / substitute) needed to turn one
    string into the other, then converts that edit distance into a
    percentage of the total length. Two identical strings need 0 edits -> 1.0.

        ratio             compares the strings as written
        token_sort_ratio  alphabetically sorts the words first

    We take max() of the two because each one fails where the other works,
    and we measured it:

        "ravi kumar" vs "kumar ravi"   ratio 0.45   token_sort 1.00   (swap)
        "ravi kumar" vs "arvi kumar"   ratio 0.90   token_sort 0.50   (typo)

    token_sort loses on the typo because sorting puts "arvi" first and
    "kumar" first, so it ends up comparing misaligned words. Using only
    token_sort_ratio - which was my first version - scored a simple typo
    lower than two unrelated names.

    The cost of max() is honest to state: taking the best of two measures
    can only push scores up, so it slightly increases false positives. That
    is a trade we make back with the thresholds in Stage 14.

    Note this is also why Stage 2 did NOT sort name tokens during
    normalization: sorting is a comparison-time choice, and here we can see
    it is not always the right one.
    """
    if not a or not b:
        return np.nan
    return max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b)) / 100.0


def partial_fuzzy_similarity(a, b):
    """
    RapidFuzz partial_ratio: the best score of `a` against any substring of
    `b` (whichever is shorter is the needle).

    Useful for abbreviations, where one string is genuinely contained in the
    other: "ravi k" inside "ravi kumar". Plain ratio punishes the missing
    letters; partial_ratio does not.
    """
    if not a or not b:
        return np.nan
    return fuzz.partial_ratio(a, b) / 100.0


# ----------------------------------------------------------------------
# B. Jaccard similarity
# ----------------------------------------------------------------------


def jaccard_similarity(a, b):
    """
    |intersection| / |union| over the SET of words.

        "12 mg road hyderabad" -> {12, mg, road, hyderabad}
        "12 mg rd hyderabad"   -> {12, mg, rd, hyderabad}

        intersection = {12, mg, hyderabad}          -> 3
        union        = {12, mg, road, rd, hyderabad} -> 5
        jaccard      = 3 / 5 = 0.6

    Properties worth knowing:
      - word order is irrelevant, so it survives reordering completely
      - it is blind to spelling: "road" and "rd" are simply different words
        (which is exactly why Stage 2 expanded abbreviations first)
      - duplicate words are ignored, because a set has no counts
    """
    if not a or not b:
        return np.nan
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a or not set_b:
        return np.nan
    return len(set_a & set_b) / len(set_a | set_b)


# ----------------------------------------------------------------------
# C. TF-IDF + cosine similarity
# ----------------------------------------------------------------------
#
# Jaccard treats every word as equally important. It should not.
# In our addresses, "road" appears in hundreds of records - two records both
# containing "road" tells us almost nothing. "kondapur" appears in a handful,
# so sharing it is real evidence.
#
# TF-IDF = Term Frequency x Inverse Document Frequency
#
#   TF(word, record)  = how often the word appears in THIS record
#   IDF(word)         = log(total records / records containing the word)
#
# IDF is the interesting half. A word in every record gets log(N/N) = 0
# weight. A word in 5 of 456 records gets log(456/5) ~ 4.5. Multiply the
# two and each record becomes a VECTOR of weighted word scores, where the
# big numbers sit on the distinctive words.
#
# Cosine similarity then measures the ANGLE between two such vectors:
#
#   cos(A, B) = (A . B) / (|A| * |B|)
#
# 1.0 means the vectors point the same way (same words, same proportions),
# 0.0 means they share no words at all. We use the angle rather than the
# distance so that a long address and a short one can still match - only
# direction matters, not magnitude.
#
# scikit-learn does the counting and the log for us. It is doing exactly the
# arithmetic above; we are not pretending it is magic.

# We build ONE vector per record from its name + address + city, so this
# feature captures overall textual agreement rather than one field.
PROFILE_COLUMNS = ["name_norm", "address_norm", "city_norm"]


def build_profile_text(df_norm):
    """Join the text fields of each record into a single 'document'."""
    return (
        df_norm[PROFILE_COLUMNS]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.strip()
    )


def build_tfidf_matrix(profile_text):
    """
    Turn the text into a sparse matrix of shape (n_records, n_vocabulary).

    min_df=1 keeps every word. norm="l2" (the default) makes every row a
    unit vector, which is what lets us compute cosine similarity as a plain
    dot product below.
    """
    vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+", min_df=1)
    matrix = vectorizer.fit_transform(profile_text)
    return vectorizer, matrix


def pairwise_cosine(matrix, rows_a, rows_b):
    """
    Cosine similarity for a LIST of pairs, not the full n x n matrix.

    sklearn's cosine_similarity(matrix, matrix) would build a 456 x 456
    array here - fine at this size, but it is the n^2 mistake Stage 4 was
    all about, and it would be 25 million cells at 5,000 records.

    Because TfidfVectorizer L2-normalizes each row, |A| = |B| = 1 and the
    cosine formula collapses to just the dot product. So we take only the
    rows we need, multiply them element-wise, and sum across.
    """
    a = matrix[rows_a]
    b = matrix[rows_b]
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def top_tfidf_terms(vectorizer, matrix, row, n=5):
    """The most distinctive words in one record - used by the dashboard."""
    vocab = np.array(vectorizer.get_feature_names_out())
    weights = matrix[row].toarray().ravel()
    order = weights.argsort()[::-1][:n]
    return [(vocab[i], round(float(weights[i]), 3)) for i in order if weights[i] > 0]


# ----------------------------------------------------------------------
# Exact agreement helpers
# ----------------------------------------------------------------------


def exact_match(a, b):
    """1.0 if both present and identical, 0.0 if both present and different."""
    if not a or not b:
        return np.nan
    return 1.0 if a == b else 0.0


if __name__ == "__main__":
    import pandas as pd

    from src.preprocessing import add_normalized_columns, load_data

    print("--- A. fuzzy vs B. jaccard on the same pairs ---")
    examples = [
        ("ravi kumar", "ravi k"),
        ("ravi kumar", "kumar ravi"),
        ("ravi kumar", "arvi kumar"),
        ("ravi kumar", "ravi sharma"),
        ("12 mg road kondapur", "12 mg road"),
        ("12 mg road kondapur", "kondapur road 12 mg"),
        ("12 mg road kondapur", "98 mg road kondapur"),
    ]
    rows = []
    for a, b in examples:
        rows.append({
            "a": a, "b": b,
            "fuzzy": round(fuzzy_similarity(a, b), 3),
            "partial": round(partial_fuzzy_similarity(a, b), 3),
            "jaccard": round(jaccard_similarity(a, b), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("""
  Read the rows:
    row 1  abbreviation    -> fuzzy 0.75 but partial 1.00 ("ravi k" is inside)
    row 2  word swap       -> fuzzy 1.00 and jaccard 1.00, both survive it
    row 3  typo            -> fuzzy 0.90 but jaccard 0.33
                             a typo creates a brand NEW token, so a
                             set-based measure collapses while a
                             character-based one barely notices
    row 4  different person-> fuzzy 0.67: usefully lower than row 3
    row 5  missing word    -> jaccard 0.75, partial 1.00
    row 7  different house -> fuzzy 0.89, jaccard 0.60, both still HIGH.
                             Same street, different house number. No string
                             measure can fix this - it is why we need the
                             phone and email features in Stage 7.
""")

    print("--- C. TF-IDF: which words carry weight? ---")
    df = add_normalized_columns(load_data("data/sample_data.csv"))
    text = build_profile_text(df)
    vec, mat = build_tfidf_matrix(text)
    print(f"  records: {mat.shape[0]}   vocabulary: {mat.shape[1]} words")

    idf = pd.DataFrame({
        "word": vec.get_feature_names_out(),
        "idf": vec.idf_,
    }).sort_values("idf")
    print("\n  lowest IDF (common, near-worthless for matching):")
    print(idf.head(5).to_string(index=False))
    print("\n  highest IDF (rare, strong evidence):")
    print(idf.tail(5).to_string(index=False))

    print(f"\n  record 0 text : {text.iloc[0]}")
    print(f"  its top terms : {top_tfidf_terms(vec, mat, 0)}")
