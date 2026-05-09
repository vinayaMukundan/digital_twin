import argparse
import re
import unicodedata
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- optional heavy deps (graceful fallback) ---
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download("stopwords",    quiet=True)
    nltk.download("wordnet",      quiet=True)
    nltk.download("punkt",        quiet=True)
    nltk.download("punkt_tab",    quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[WARN] nltk not installed — lemmatization / stopword removal disabled.")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

_RE_URL        = re.compile(r"https?://\S+|www\.\S+")
_RE_HTML       = re.compile(r"<[^>]+>")
_RE_EMAIL      = re.compile(r"\S+@\S+\.\S+")
_RE_MENTION    = re.compile(r"@\w+")
_RE_HASHTAG    = re.compile(r"#\w+")
_RE_NON_ASCII  = re.compile(r"[^\x00-\x7F]+")
_RE_SPECIAL    = re.compile(r"[^a-zA-Z0-9\s'.,!?]") 
_RE_EXTRA_WS   = re.compile(r"\s{2,}")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = _RE_URL.sub("", text)
    text = _RE_HTML.sub("", text)
    text = _RE_EMAIL.sub("", text)
    text = _RE_MENTION.sub("", text)
    text = _RE_HASHTAG.sub("", text)
    text = _RE_NON_ASCII.sub("", text)
    text = _RE_SPECIAL.sub(" ", text)
    text = _RE_EXTRA_WS.sub(" ", text).strip()
    return text

# ... (Tokenization and Lemmatization functions remain the same) ...

def tokenize_text(text: str) -> list[str]:
    if NLTK_AVAILABLE: return nltk.word_tokenize(text)
    return text.split()

def remove_stopwords(tokens: list[str]) -> list[str]:
    if not NLTK_AVAILABLE: return tokens
    sw = set(stopwords.words("english"))
    return [t for t in tokens if t not in sw]

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    if not NLTK_AVAILABLE: return tokens
    lem = WordNetLemmatizer()
    return [lem.lemmatize(t) for t in tokens]

def encode_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series):
        return series.astype(int)
    unique_vals = sorted(series.dropna().unique())
    mapping = {v: i for i, v in enumerate(unique_vals)}
    print(f"[Labels] Encoding map: {mapping}")
    return series.map(mapping)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(
    input_path: str,
    output_path: str,
    text_col: str,
    label_col: str,
    apply_lemma: bool,
    apply_stopwords: bool,
    min_token_length: int,
) -> pd.DataFrame:

    print(f"\n{'='*60}")
    print("  Mental Health Corpus — Preprocessing Pipeline")
    print(f"{'='*60}")

    # Verify input exists
    if not os.path.exists(input_path):
        # Fallback logic: if running from inside 'src', look up one level
        if not input_path.startswith("/") and not os.path.exists(input_path):
            input_path = os.path.join("..", input_path)
            output_path = os.path.join("..", output_path)

    print(f"[1/6] Loading '{input_path}' ...")
    df = pd.read_csv(input_path)
    
    # Returns (rows, columns)
    print(df.shape) 

    # Returns only rows
    row_count = df.shape[0]
    print(row_count)
    # pd.set_option('display.max_colwidth', None)
   
    # --- NEW: Print first 20 rows of raw data ---
    # print("\n>>> FIRST 20 ROWS OF RAW DATA:")
    # print(df.head(20))
    # print("-" * 30)
    
    # ... (Processing logic: Handling missing, cleaning, filtering) ...
    df.dropna(subset=[text_col, label_col], inplace=True)
    df["cleaned_text"] = df[text_col].astype(str).apply(clean_text)
    df = df[df["cleaned_text"].str.strip() != ""]

    if apply_stopwords or apply_lemma:
        tokens_col = df["cleaned_text"].apply(tokenize_text)
        if apply_stopwords: tokens_col = tokens_col.apply(remove_stopwords)  # noqa: E701
        if apply_lemma: tokens_col = tokens_col.apply(lemmatize_tokens)  # noqa: E701
        df["cleaned_text"] = tokens_col.apply(lambda t: " ".join(t))

    df = df[df["cleaned_text"].str.split().str.len() >= min_token_length]
    df["encoded_label"] = encode_labels(df[label_col])

    # Save output
    print(f"[6/6] Saving to '{output_path}' ...")
    out_df = df[["cleaned_text", "encoded_label"]].rename(
        columns={"cleaned_text": "text", "encoded_label": "label"}
    )
    
    
    # --- NEW: Print first 20 rows of preprocessed data ---
    # print("\n>>> FIRST 20 ROWS OF PREPROCESSED DATA:")
    # print(out_df.head(20))
    # print("-" * 30)
    
    # NEW: Splitting Logic
    # 70% Train, 15% Validation, 15% Test
    train_df, temp_df = train_test_split(out_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Save outputs
    data_dir = os.path.dirname(output_path)
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    
    print(f"Saved: train.csv ({len(train_df)}), val.csv ({len(val_df)}), test.csv ({len(test_df)})")

if __name__ == "__main__":
    # Hardcoded paths relative to the project root
    # This means you can just run 'python src/preprocess.py'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_INPUT = os.path.join(BASE_DIR, "data", "mental_health.csv")
    DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data", "preprocessed.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",            default=DEFAULT_INPUT)
    parser.add_argument("--output",           default=DEFAULT_OUTPUT)
    parser.add_argument("--text_col",         default="text")
    parser.add_argument("--label_col",        default="label")
    parser.add_argument("--apply_lemma",      default="False")
    parser.add_argument("--apply_stopwords",  default="False")
    parser.add_argument("--min_token_length", type=int, default=3)

    args = parser.parse_args()

    preprocess(
        input_path        = args.input,
        output_path       = args.output,
        text_col          = args.text_col,
        label_col         = args.label_col,
        apply_lemma       = str(args.apply_lemma).lower() == "true",
        apply_stopwords   = str(args.apply_stopwords).lower() == "true",
        min_token_length  = args.min_token_length,
    )