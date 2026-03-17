import math
import json
from collections import defaultdict
from typing import Optional

QUERY_SEP = "\x18"
LIST_SEP = "_"
NO_QUERY = "-1"

SPECIAL_TOKENS = {
    "[PAD]":      0,
    "[UNK]":      1,
    "[NO_QUERY]": 2,
    "[MASK]":     3,
}

INTERACTION_MAP = {
    "ORD":   3,   # purchased
    "CART":  2,   # added to cart
    "CLICK": 1,   # clicked
    "[PAD]": 0,   # padding
}

class Vocabulary:
    """
    Builds and maintains a mapping from raw term/product/shop IDs
    to dense integer indices suitable for embedding lookup.
    """

    def __init__(self):
        self.token2idx: dict[str, int] = dict(SPECIAL_TOKENS)
        self.idx2token: dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        self._next_idx: int = len(SPECIAL_TOKENS)


    def add_token(self, token: str) -> int:
        """Add a token if unseen; return its index either way."""
        token = str(token)
        if token not in self.token2idx:
            self.token2idx[token] = self._next_idx
            self.idx2token[self._next_idx] = token
            self._next_idx += 1
        return self.token2idx[token]

    def build_from_user_records(self, records: list[dict]) -> None:
        """
        Scan all user behavior records and register every term ID found in:
            - query	
            - candidate_wid_list
            - history_qry_list
            - history_wid_list
        """
        for rec in records:
            # query terms
            for term in parse_list(rec.get("query", ""), QUERY_SEP):
                self.add_token(term)

            # history query terms
            for q in parse_query_list(rec.get("history_qry_list", "")):
                for term in q:
                    self.add_token(term)

            # product ids (history + candidates)
            for hwl in parse_list(rec.get("history_wid_list", ""), LIST_SEP):
                self.add_token(hwl)
            for cwl in parse_list(rec.get("candidate_wid_list", ""), LIST_SEP):
                self.add_token(cwl)


    def build_from_product_records(self, records: list[dict]) -> None:
        """
        Scan all product records and register every term/ID found in:
          - wid, shop_id, brand_id
          - name, brand_name, category_name_{1..4}
          - category_id_{1..4}
        """
        for rec in records:
            for field in ["wid", "shop_id", "brand_id",
                          "cate_id_1", "cate_id_2",
                          "cate_id_3", "cate_id_4"]:
                val = rec.get(field)
                if val is not None:
                    self.add_token(str(val))

            for field in ["name", "brand_name",
                          "cate_name_1", "cate_name_2",
                          "cate_name_3", "cate_name_4"]:
                for term in parse_list(rec.get(field, ""), QUERY_SEP):
                    self.add_token(term)

    def encode(self, token: str) -> int:
        """Return vocab index for token; fall back to [UNK] if unseen."""
        return self.token2idx.get(str(token), self.token2idx["[UNK]"])

    def encode_sequence(self, tokens: list[str]) -> list[int]:
        return [self.encode(t) for t in tokens]
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.token2idx, f)
        print(f"Vocabulary saved → {path}  ({len(self.token2idx)} tokens)")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = cls()
        with open(path) as f:
            token2idx = json.load(f)
        vocab.token2idx = token2idx
        vocab.idx2token = {v: k for k, v in token2idx.items()}
        vocab._next_idx = max(token2idx.values()) + 1
        return vocab

    def __len__(self) -> int:
        return len(self.token2idx)

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"

# ─────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────

def parse_list(raw: str, sep: str) -> list[str]:
    """
    Split a single query string by the specified separator.
    """
    if not raw or str(raw).strip() == "":
        return []
    return str(raw).strip().split(sep)


def parse_float_list(raw: str, sep: str) -> list[float]:
    """
    '3.0_0.0_0.0'  →  [3.0, 0.0, 0.0]
    """
    if not raw or str(raw).strip() == "":
        return []
    return [float(x) for x in str(raw).strip().split(sep)]


def parse_int_list(raw: str, sep: str) -> list[int]:
    """
    '0_12_4_5'  →  [0, 12, 4, 5]
    """
    if not raw or str(raw).strip() == "":
        return []
    return [int(x) for x in str(raw).strip().split(sep)]


def parse_query_list(raw: str) -> list[list[str]]:
    """
    Parse a _-separated list of queries, where each query may itself
    contain \x18 -separated terms.

    '323\x18328\x18196_-1_12\x1854'
        →  [['323','328','196'], ['-1'], ['12','54']]
    """
    if not raw or str(raw).strip() == "":
        return []
    queries = []
    for q in str(raw).strip().split(LIST_SEP):
        queries.append(parse_list(q, QUERY_SEP))
    return queries


def encode_interactions(actions: list[str]) -> list[int]:
    """
    Map interaction strings to integers.
    Unknown strings fall back to 0 (same as [PAD]).

    ['ORD', 'CLICK', 'CART']  →  [3, 1, 2]
    """
    return [INTERACTION_MAP.get(a, 0) for a in actions]


def log_normalize_time(intervals: list[int]) -> list[float]:
    """
    Apply log(1 + x) to compress large time gaps.

    [0, 12, 4, 5]  →  [0.0, 2.565, 1.609, 1.792]
    """
    return [round(math.log1p(t), 4) for t in intervals]

def binarize_labels(labels: list[float]) -> list[int]:
    """
    Convert graded labels to binary: purchased (>0) → 1, else → 0.

    [3.0, 0.0, 0.0]  →  [1, 0, 0]
    """
    return [1 if l > 0 else 0 for l in labels]


def pad_or_truncate(seq: list, max_len: int, pad_value=0) -> list:
    """
    Truncate sequence to max_len (keeping the earliest items),
    then right-pad with pad_value up to max_len.

    Example (max_len=5):
        [1, 2, 3]          →  [1, 2, 3, 0, 0]
        [1, 2, 3, 4, 5, 6] →  [1, 2, 3, 4, 5]   (newest dropped)
    """
    seq = seq[:max_len]                          # truncate from the right
    seq = seq + [pad_value] * (max_len - len(seq))  # right-pad
    return seq

# ─────────────────────────────────────────
# FILE-LEVEL PROCESSORS
# ─────────────────────────────────────────

def load_tsv(filepath: str) -> list[dict]:
    """Read a tab-separated file into a list of dicts using the header row."""
    records = []
    with open(filepath, "r") as f:
        headers = f.readline().strip().split("\t")
        for line in f:
            values = line.strip().split("\t")
            if len(values) == len(headers):
                records.append(dict(zip(headers, values)))
    print(f"Loaded {len(records)} records from {filepath}")
    return records


def process_user_file(
    filepath: str,
    vocab: Vocabulary,
    max_query_len: int   = 10,
    max_history_len: int = 20,
) -> list[dict]:
    """
    Load and process an entire user behavior TSV file.

    Steps:
        1. Load raw records from TSV
        2. Update vocabulary with all term IDs found
        3. Process each record into model-ready format

    Returns list of processed user dicts.
    """
    records = load_tsv(filepath)

    print("Building vocabulary from user records...")
    vocab.build_from_user_records(records)

    print("Processing user records...")
    processed = []
    skipped   = 0
    for i, rec in enumerate(records):
        try:
            processed.append(
                process_user_record(rec, vocab, max_query_len, max_history_len)
            )
        except AssertionError as e:
            print(f"  [SKIP] Record {i} failed alignment check: {e}")
            skipped += 1

    print(f"Done. {len(processed)} processed, {skipped} skipped.")
    return processed


def process_product_file(
    filepath: str,
    vocab: Vocabulary,
    max_name_len: int     = 10,
    max_brand_len: int    = 5,
    max_cat_name_len: int = 5,
) -> dict[int, dict]:
    """
    Load and process an entire product TSV file.

    Steps:
        1. Load raw records from TSV
        2. Update vocabulary with all term IDs found
        3. Process each record into model-ready format

    Returns dict mapping raw wid (int) → processed product dict,
    for fast lookup during training (user's candidate products → item features).
    """
    records = load_tsv(filepath)

    print("Building vocabulary from product records...")
    vocab.build_from_product_records(records)

    print("Processing product records...")
    product_lookup = {}
    for rec in records:
        processed = process_product_record(
            rec, vocab, max_name_len, max_brand_len, max_cat_name_len
        )
        raw_wid = int(rec.get("wid", -1))
        product_lookup[raw_wid] = processed

    print(f"Done. {len(product_lookup)} products processed.")
    return product_lookup


# ─────────────────────────────────────────
# USER RECORD PROCESSING
# ─────────────────────────────────────────

def process_user_record(
    record: dict,
    vocab: Vocabulary,
    max_query_len: int  = 10,
    max_history_len: int = 20,
) -> dict:
    """
    Process a single raw user behavior record into model-ready tensors.

    Parameters
    ----------
    record          : dict with keys matching the user behavior TSV columns
    vocab           : fitted Vocabulary instance
    max_query_len   : max number of terms in any single query
    max_history_len : max number of historical interactions to keep

    Returns
    -------
    dict with all fields padded, encoded, and ready for the User Tower.

    Example input
    -------------
    {
        "query":                    "12^X32^X56",
        "candidate_wid_list":       "456_457_789",
        "candidate_label_list":     "3.0_0.0_0.0",
        "history_qry_list":         "323^X328^X196_-1_12^X54",
        "history_wid_list":         "889_256_345",
        "history_type_list":        "ORD_CLICK_CART",
        "history_time_list":        "0_12_4_5"
    }
    """

    # ── 1. Parse raw strings ──────────────────────────────────────────────────

    query_terms        = parse_list(record.get("query", ""), QUERY_SEP)
    candidate_products = parse_list(record.get("candidate_wid_list", ""), LIST_SEP)
    raw_labels         = parse_float_list(record.get("candidate_label_list", ""), LIST_SEP)
    history_query_list = parse_query_list(record.get("history_qry_list", ""))
    history_products   = parse_list(record.get("history_wid_list", ""), LIST_SEP)
    history_actions    = parse_list(record.get("history_type_list", ""), LIST_SEP)
    time_intervals     = parse_int_list(record.get("history_time_list", ""), LIST_SEP)

    # ── 2. Validate alignment ────────────────────────────────────────────────
    n_history = len(history_products)
    assert len(history_query_list) == n_history, (
        f"history_queries length {len(history_query_list)} != "
        f"history_products length {n_history}"
    )
    assert len(history_actions) == n_history, (
        f"history_actions length {len(history_actions)} != "
        f"history_products length {n_history}"
    )

    # ── 3. Encode query ──────────────────────────────────────────────────
    query_ids = vocab.encode_sequence(query_terms)
    query_ids = pad_or_truncate(query_ids, max_query_len)

    # ── 4. Encode history queries ─────────────────────────────────────────────
    encoded_history_queries = []
    for q_terms in history_query_list:
        # Replace -1 sentinel with [NO_QUERY] token
        if q_terms == [NO_QUERY]:
            q_ids = [vocab.encode("[NO_QUERY]")]
        else:
            q_ids = vocab.encode_sequence(q_terms)
        q_ids = pad_or_truncate(q_ids, max_query_len)
        encoded_history_queries.append(q_ids)

    # Pad/truncate the history sequence itself
    pad_query = [0] * max_query_len
    encoded_history_queries = pad_or_truncate(
        encoded_history_queries, max_history_len, pad_value=pad_query
    )

    # ── 5. Encode history products ────────────────────────────────────────────
    history_product_ids = vocab.encode_sequence(history_products)
    history_product_ids = pad_or_truncate(history_product_ids, max_history_len)

    # ── 6. Encode interaction levels ──────────────────────────────────────────
    interaction_ids = encode_interactions(history_actions)
    interaction_ids = pad_or_truncate(interaction_ids, max_history_len)

    # ── 7. Normalize time intervals ───────────────────────────────────────────
    norm_time = log_normalize_time(time_intervals)
    norm_time = pad_or_truncate(norm_time, max_history_len + 1, pad_value=0.0)
    # +1 because time_intervals includes the gap to the query

    # ── 8. Encode labels ──────────────────────────────────────────────────────
    binary_labels = binarize_labels(raw_labels)

    return {
        # Query
        "query":           query_ids,

        # Candidate products + labels
        "candidate_products":   [vocab.encode(p) for p in candidate_products],
        "labels":               binary_labels,

        # History queries
        "history_queries":      encoded_history_queries,

        # History products
        "history_products":     history_product_ids,

        # Interaction levels
        "history_actions":      interaction_ids,

        # Time intervals (log-normalized)
        "time_intervals":       norm_time,
    }

# ─────────────────────────────────────────
# PRODUCT RECORD PROCESSING
# ─────────────────────────────────────────

def process_product_record(
    record: dict,
    vocab: Vocabulary,
    max_name_len: int     = 10,
    max_brand_len: int    = 5,
    max_cat_name_len: int = 5,
) -> dict:
    """
    Process a single raw product record into model-ready tensors.

    Parameters
    ----------
    record           : dict with keys matching the product TSV columns
    vocab            : fitted Vocabulary instance
    max_name_len     : max terms in product name
    max_brand_len    : max terms in brand name
    max_cat_name_len : max terms per category name level

    Returns
    -------
    dict with all fields encoded, padded for the Item Tower.

    Example input
    -------------
    {
        "wid":            "456",
        "name":           "78^X91^X204",
        "brand_id":       "33",
        "brand_name":     "501^X502",
        "cate_id_1":      "10",
        "cate_id_2":      "47",
        "cate_id_3":      "112",
        "cate_id_4":      "189",
        "cate_name_1":    "601",
        "cate_name_2":    "612^X615",
        "cate_name_3":    "634",
        "cate_name_4":    "641^X643",
        "shop_id":        "77"
    }
    """

    # ── 1. Product ID ─────────────────────────────────────────────────────────
    wid = vocab.encode(record.get("wid", "[UNK]"))

    # ── 2. Product name ───────────────────────────────────────────────────────
    name_terms = parse_list(record.get("name", ""), QUERY_SEP)
    name_ids   = vocab.encode_sequence(name_terms)
    name_ids   = pad_or_truncate(name_ids, max_name_len)

    # ── 3. Brand ──────────────────────────────────────────────────────────────
    brand_id = vocab.encode(record.get("brand_id", "[UNK]"))

    brand_name_terms = parse_list(record.get("brand_name", ""), QUERY_SEP)
    brand_name_ids   = vocab.encode_sequence(brand_name_terms)
    brand_name_ids   = pad_or_truncate(brand_name_ids, max_brand_len)

    # ── 4. Category IDs (4 levels, no padding needed — always 4 values) ───────
    category_ids = []
    for level in range(1, 5):
        raw = record.get(f"cate_id_{level}")
        category_ids.append(
            vocab.encode(str(raw)) if raw is not None else vocab.encode("[UNK]")
        )

    # ── 5. Category names (4 levels, each padded independently) ──────────────
    category_name_ids   = []
    for level in range(1, 5):
        raw_name  = record.get(f"cate_name_{level}", "")
        terms     = parse_list(raw_name, QUERY_SEP)
        ids       = vocab.encode_sequence(terms)
        ids       = pad_or_truncate(ids, max_cat_name_len)
        category_name_ids.append(ids)

    # ── 6. Shop ID ────────────────────────────────────────────────────────────
    shop_id = vocab.encode(record.get("shop_id", "[UNK]"))

    return {
        "wid":                wid,

        "name":               name_ids,

        "brand_id":           brand_id,
        "brand_name":         brand_name_ids,

        "category_ids":       category_ids,        # [lvl1, lvl2, lvl3, lvl4]
        "category_names":     category_name_ids,   # [[...], [...], [...], [...]]

        "shop_id":            shop_id,
    }

# ─────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────

if __name__ == "__main__":

    # ── Load and process real data files ──────────────────────────────────────
    user_filepath = "/Users/Akshita/Downloads/archive/user_behavior_data.txt"
    product_filepath = "/Users/Akshita/Downloads/archive/product_meta_data.txt"

    vocab = Vocabulary()
    processed_users = process_user_file(user_filepath, vocab)
    processed_products = process_product_file(product_filepath, vocab)

    # ── Print example processed records (first user and one product) ─────────
    if processed_users:
        print("\n=== PROCESSED USER RECORD (first one) ===")
        for k, v in processed_users[0].items():
            print(f"  {k:30s}: {v}")

    if processed_products:
        example_wid = next(iter(processed_products.keys()))
        print(f"\n=== PROCESSED PRODUCT RECORD (wid={example_wid}) ===")
        for k, v in processed_products[example_wid].items():
            print(f"  {k:30s}: {v}")