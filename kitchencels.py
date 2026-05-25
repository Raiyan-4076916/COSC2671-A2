from __future__ import annotations

import json
import re
import os
import hashlib
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# ============================================================
# CONFIG
# ============================================================

POSTS_FILE = Path("r_kitchencels_posts.jsonl")
COMMENTS_FILE = Path("r_kitchencels_comments.jsonl")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
FIG_DIR = Path("figures")

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

SUBREDDIT_NAME = "kitchencels"

HASH_SALT = os.getenv("HASH_SALT", "kitchencels_assignment_salt_2026")

# Use all comments by default. Set to an integer if testing.
MAX_COMMENTS_TO_READ = None

# Topic modelling can become slow on huge datasets.
MAX_LDA_DOCS = 25000

# Transformer sentiment can be slow on CPU.
# Use None to run on all items, or set an integer such as 20000.
USE_ROBERTA_SENTIMENT = True
MAX_ROBERTA_ITEMS = None
ROBERTA_BATCH_SIZE = 32

# Zero-shot classification is much slower, so it should be run on a sample.
USE_ZERO_SHOT = True
ZERO_SHOT_SAMPLE_SIZE = 600
ZERO_SHOT_BATCH_SIZE = 8

# Keep VADER only as a baseline comparison.
USE_VADER_BASELINE = True

RANDOM_STATE = 42


# ============================================================
# NLTK SETUP
# ============================================================

for pkg in ["stopwords", "wordnet", "vader_lexicon", "punkt"]:
    nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))

CUSTOM_STOPWORDS = {
    "http", "https", "www", "com", "reddit", "removed", "deleted",
    "would", "could", "like", "get", "got", "one", "thing", "things",
    "really", "even", "much", "make", "made", "know", "see", "say",
    "said", "people", "time", "still", "also", "back", "going",
    "want", "need", "think", "look", "way", "post", "comment",
    "subreddit", "thread", "op"
}

STOPWORDS = STOPWORDS | CUSTOM_STOPWORDS
LEMMATIZER = WordNetLemmatizer()


# ============================================================
# HELPERS
# ============================================================

def save_fig(name: str):
    plt.tight_layout()
    path = FIG_DIR / name
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def parse_utc(value):
    if value is None or value == "":
        return pd.NaT

    try:
        return pd.to_datetime(float(value), unit="s", utc=True)
    except Exception:
        pass

    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return pd.NaT


def anonymise_author(author) -> str:
    if author is None:
        return "unknown"

    author = str(author).strip()

    if author == "" or author.lower() in {
        "[deleted]", "deleted", "automoderator", "none", "nan"
    }:
        return "unknown"

    raw = f"{HASH_SALT}:{author}".encode("utf-8")
    return "user_" + hashlib.sha256(raw).hexdigest()[:10]


URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"&amp;|&lt;|&gt;|&quot;|&#x200b;|&apos;")
NON_WORD_RE = re.compile(r"[^a-zA-Z\s']+")


def clean_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = text.replace("\n", " ")
    text = NON_WORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    text = clean_text(text)
    words = re.findall(r"[a-z][a-z']{2,}", text)

    tokens = []
    for w in words:
        if w in STOPWORDS:
            continue
        if len(w) < 3:
            continue
        tokens.append(LEMMATIZER.lemmatize(w))

    return tokens


def safe_get(obj: dict, key: str, default=None):
    value = obj.get(key, default)
    if value is None:
        return default
    return value


def read_jsonl_selected(path: Path, kind: str, limit=None) -> pd.DataFrame:
    rows = []

    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"Reading {kind}")):
            if limit is not None and i >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if kind == "post":
                rows.append({
                    "id": safe_get(obj, "id"),
                    "subreddit": safe_get(obj, "subreddit"),
                    "author": safe_get(obj, "author"),
                    "created_utc": safe_get(obj, "created_utc"),
                    "title": safe_get(obj, "title", ""),
                    "selftext": safe_get(obj, "selftext", ""),
                    "score": safe_get(obj, "score", 0),
                    "num_comments": safe_get(obj, "num_comments", 0),
                    "link_flair_text": safe_get(obj, "link_flair_text", ""),
                    "over_18": safe_get(obj, "over_18", False),
                    "spoiler": safe_get(obj, "spoiler", False),
                    "is_self": safe_get(obj, "is_self", False),
                    "is_video": safe_get(obj, "is_video", False),
                    "domain": safe_get(obj, "domain", ""),
                    "url": safe_get(obj, "url", ""),
                    "permalink": safe_get(obj, "permalink", ""),
                    "removed_by_category": safe_get(obj, "removed_by_category", ""),
                    "post_hint": safe_get(obj, "post_hint", ""),
                })

            elif kind == "comment":
                rows.append({
                    "id": safe_get(obj, "id"),
                    "subreddit": safe_get(obj, "subreddit"),
                    "author": safe_get(obj, "author"),
                    "created_utc": safe_get(obj, "created_utc"),
                    "body": safe_get(obj, "body", ""),
                    "score": safe_get(obj, "score", 0),
                    "link_id": safe_get(obj, "link_id", ""),
                    "parent_id": safe_get(obj, "parent_id", ""),
                    "permalink": safe_get(obj, "permalink", ""),
                    "distinguished": safe_get(obj, "distinguished", ""),
                    "controversiality": safe_get(obj, "controversiality", 0),
                })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df):,} {kind}s")
    return df


def normalise_hf_sentiment_label(label: str) -> str:
    label = str(label).lower()

    mapping = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }

    return mapping.get(label, label)


# ============================================================
# LOAD DATA
# ============================================================

posts = read_jsonl_selected(POSTS_FILE, "post")
comments = read_jsonl_selected(COMMENTS_FILE, "comment", limit=MAX_COMMENTS_TO_READ)

posts = posts.dropna(subset=["id"]).drop_duplicates("id")
comments = comments.dropna(subset=["id"]).drop_duplicates("id")

posts["created_dt"] = posts["created_utc"].apply(parse_utc)
comments["created_dt"] = comments["created_utc"].apply(parse_utc)

posts["created_date"] = posts["created_dt"].dt.date
comments["created_date"] = comments["created_dt"].dt.date

posts["month"] = posts["created_dt"].dt.strftime("%Y-%m")
comments["month"] = comments["created_dt"].dt.strftime("%Y-%m")

posts["author_anon"] = posts["author"].apply(anonymise_author)
comments["author_anon"] = comments["author"].apply(anonymise_author)

posts["text"] = (
    posts["title"].fillna("").astype(str)
    + " "
    + posts["selftext"].fillna("").astype(str)
).str.strip()

comments["text"] = comments["body"].fillna("").astype(str)

posts = posts[~posts["text"].str.strip().isin(["", "[deleted]", "[removed]"])]
comments = comments[~comments["text"].str.strip().isin(["", "[deleted]", "[removed]"])]

posts["item_type"] = "post"
comments["item_type"] = "comment"

items = pd.concat([
    posts[[
        "id", "item_type", "subreddit", "author_anon", "created_dt",
        "created_date", "month", "text", "score", "num_comments",
        "link_flair_text", "permalink"
    ]],
    comments[[
        "id", "item_type", "subreddit", "author_anon", "created_dt",
        "created_date", "month", "text", "score", "permalink"
    ]].assign(num_comments=0, link_flair_text="")
], ignore_index=True)

items["clean_text"] = items["text"].apply(clean_text)
items["tokens"] = items["clean_text"].apply(tokenize)
items["token_text"] = items["tokens"].apply(lambda x: " ".join(x))


# ============================================================
# CUSTOM THEMES AND AFFECTIVE LEXICONS
# ============================================================

THEME_LEXICONS = {
    "food_and_cooking": [
        "cook", "cooking", "meal", "food", "eat", "eating", "dinner",
        "lunch", "breakfast", "kitchen", "plate", "recipe", "rice",
        "chicken", "pasta", "soup", "egg", "meat", "bread", "cheese",
        "vegetable", "sauce", "airfryer", "microwave", "oven", "stove",
        "noodle", "ramen", "sandwich", "pizza", "burger", "beans",
        "potato", "salad", "curry", "steak", "fish", "tuna"
    ],
    "food_shame_or_slop": [
        "slop", "disgusting", "burnt", "raw", "undercooked", "overcooked",
        "microwave meal", "instant", "struggle meal", "sad meal",
        "poverty meal", "looks bad", "tastes bad", "inedible", "gross"
    ],
    "loneliness_and_isolation": [
        "alone", "lonely", "loneliness", "isolated", "isolation",
        "friendless", "no friends", "nobody", "single", "touch starved",
        "ignored", "invisible", "outcast", "left out", "no one cares"
    ],
    "dating_and_rejection": [
        "date", "dating", "girl", "girlfriend", "boyfriend", "crush",
        "rejected", "rejection", "relationship", "tinder", "match",
        "love", "kiss", "virgin", "sex", "romantic", "wife", "husband",
        "ghosted", "asked out", "friendzone"
    ],
    "self_deprecation_and_shame": [
        "ugly", "loser", "failure", "fail", "pathetic", "shame",
        "embarrassing", "humiliation", "humiliated", "worthless",
        "subhuman", "over for me", "it's over", "it is over", "cooked",
        "i am done", "i'm done"
    ],
    "distress_and_despair_language": [
        "depressed", "depression", "anxiety", "anxious", "sad",
        "suicide", "suicidal", "kill myself", "want to die", "die",
        "death", "hopeless", "tired of life", "cry", "crying",
        "pain", "suffer", "suffering", "can't take it", "cannot take it"
    ],
    "work_school_and_status": [
        "job", "work", "school", "college", "class", "exam",
        "study", "student", "teacher", "boss", "shift", "money",
        "poor", "rich", "status", "neet", "wage", "wages",
        "unemployed", "minimum wage"
    ],
    "community_slang_and_meta": [
        "kitchencel", "kitchencels", "incel", "cel", "truecel",
        "fakecel", "chad", "stacy", "foid", "moid", "normie",
        "cope", "maxx", "looksmaxx", "billions must", "must cook",
        "platemogging", "platemogged", "mog", "mogged"
    ],
    "irony_or_meme_performance": [
        "billions must", "must cook", "platemogging", "platemogged",
        "mogged", "mog", "fakecel", "truecel", "cope", "it's over",
        "it is over", "over for", "cooked", "many such cases",
        "brutal", "never began"
    ],
    "gendered_resentment_language": [
        "foid", "moid", "stacy", "chad", "women", "girls", "men",
        "female", "male", "dating market", "hypergamy"
    ],
}


def count_phrase_matches(text: str, phrases: list[str]) -> int:
    text_l = clean_text(text)
    count = 0

    for phrase in phrases:
        phrase_l = phrase.lower()

        if " " in phrase_l:
            count += text_l.count(phrase_l)
        else:
            count += len(re.findall(rf"\b{re.escape(phrase_l)}\b", text_l))

    return count


for category, phrases in THEME_LEXICONS.items():
    count_col = f"{category}_count"
    flag_col = f"{category}_flag"

    items[count_col] = items["text"].apply(lambda x: count_phrase_matches(str(x), phrases))
    items[flag_col] = (items[count_col] > 0).astype(int)

THEME_FLAG_COLS = [f"{x}_flag" for x in THEME_LEXICONS.keys()]
THEME_COUNT_COLS = [f"{x}_count" for x in THEME_LEXICONS.keys()]

items["n_theme_flags"] = items[THEME_FLAG_COLS].sum(axis=1)


def dominant_theme(row) -> str:
    scores = {theme: row[f"{theme}_count"] for theme in THEME_LEXICONS.keys()}
    scores = {k: v for k, v in scores.items() if v > 0}

    if not scores:
        return "uncategorised"

    return max(scores, key=scores.get)


items["dominant_theme"] = items.apply(dominant_theme, axis=1)

theme_summary = []

for theme in THEME_LEXICONS.keys():
    theme_summary.append({
        "theme": theme,
        "items_with_theme": int(items[f"{theme}_flag"].sum()),
        "total_keyword_matches": int(items[f"{theme}_count"].sum()),
        "percentage_of_items": float(100 * items[f"{theme}_flag"].sum() / len(items))
    })

theme_summary_df = pd.DataFrame(theme_summary).sort_values(
    "items_with_theme",
    ascending=False
)

theme_summary_df.to_csv(OUTPUT_DIR / "custom_affective_theme_summary.csv", index=False)


# ============================================================
# BASELINE VADER SENTIMENT
# ============================================================

if USE_VADER_BASELINE:
    sia = SentimentIntensityAnalyzer()

    items["vader_compound"] = items["text"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    def vader_label(score: float) -> str:
        if score >= 0.05:
            return "positive"
        if score <= -0.05:
            return "negative"
        return "neutral"

    items["vader_sentiment_label"] = items["vader_compound"].apply(vader_label)
else:
    items["vader_compound"] = np.nan
    items["vader_sentiment_label"] = "not_run"


# ============================================================
# ROBERTA SOCIAL-MEDIA SENTIMENT
# ============================================================

items["roberta_sentiment_label"] = np.nan
items["roberta_sentiment_score"] = np.nan

if USE_ROBERTA_SENTIMENT:
    try:
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1

        print("\nLoading RoBERTa sentiment model...")
        print("Device:", "GPU" if device == 0 else "CPU")

        roberta_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device
        )

        if MAX_ROBERTA_ITEMS is not None and len(items) > MAX_ROBERTA_ITEMS:
            roberta_items = items.sample(MAX_ROBERTA_ITEMS, random_state=RANDOM_STATE).copy()
            print(f"Running RoBERTa on sample of {len(roberta_items):,} items")
        else:
            roberta_items = items.copy()
            print(f"Running RoBERTa on all {len(roberta_items):,} items")

        texts = roberta_items["text"].fillna("").astype(str).tolist()
        ids = roberta_items["id"].tolist()

        roberta_rows = []

        for i in tqdm(range(0, len(texts), ROBERTA_BATCH_SIZE), desc="RoBERTa sentiment"):
            batch_texts = texts[i:i + ROBERTA_BATCH_SIZE]
            batch_ids = ids[i:i + ROBERTA_BATCH_SIZE]

            preds = roberta_pipe(
                batch_texts,
                truncation=True,
                max_length=256
            )

            for item_id, pred in zip(batch_ids, preds):
                roberta_rows.append({
                    "id": item_id,
                    "roberta_sentiment_label": normalise_hf_sentiment_label(pred["label"]),
                    "roberta_sentiment_score": float(pred["score"])
                })

        roberta_df = pd.DataFrame(roberta_rows)

        roberta_df.to_csv(
            OUTPUT_DIR / "roberta_sentiment_item_predictions.csv",
            index=False
        )

        items = items.merge(
            roberta_df,
            on="id",
            how="left",
            suffixes=("", "_new")
        )

        items["roberta_sentiment_label"] = items["roberta_sentiment_label_new"].combine_first(
            items["roberta_sentiment_label"]
        )

        items["roberta_sentiment_score"] = items["roberta_sentiment_score_new"].combine_first(
            items["roberta_sentiment_score"]
        )

        items = items.drop(
            columns=["roberta_sentiment_label_new", "roberta_sentiment_score_new"],
            errors="ignore"
        )

        roberta_summary = (
            roberta_df["roberta_sentiment_label"]
            .value_counts()
            .reset_index()
        )

        roberta_summary.columns = ["sentiment_label", "count"]

        roberta_summary["percentage"] = (
            100 * roberta_summary["count"] / roberta_summary["count"].sum()
        )

        roberta_summary.to_csv(
            OUTPUT_DIR / "roberta_sentiment_summary.csv",
            index=False
        )

    except Exception as e:
        warnings.warn(f"RoBERTa sentiment skipped because of error: {e}")


# ============================================================
# ZERO-SHOT THEME CLASSIFICATION SAMPLE
# ============================================================

items["zero_shot_top_label"] = np.nan
items["zero_shot_top_score"] = np.nan

if USE_ZERO_SHOT:
    try:
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1

        print("\nLoading zero-shot classification model...")
        print("Device:", "GPU" if device == 0 else "CPU")

        zero_shot_pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device
        )

        candidate_labels = [
            "ordinary cooking or food discussion",
            "food shame or cooking failure",
            "loneliness and social isolation",
            "self-deprecating humour",
            "dating rejection or romantic frustration",
            "mental distress or hopelessness",
            "community meme or slang performance",
            "work or school status anxiety",
            "gendered resentment language"
        ]

        zero_shot_sample = items.sample(
            min(ZERO_SHOT_SAMPLE_SIZE, len(items)),
            random_state=RANDOM_STATE
        ).copy()

        zs_rows = []

        for row in tqdm(
            zero_shot_sample.itertuples(index=False),
            total=len(zero_shot_sample),
            desc="Zero-shot classification"
        ):
            text = str(row.text)
            text_for_model = text[:1000]

            result = zero_shot_pipe(
                text_for_model,
                candidate_labels=candidate_labels,
                multi_label=True
            )

            zs_rows.append({
                "id": row.id,
                "item_type": row.item_type,
                "dominant_theme_lexicon": row.dominant_theme,
                "text_excerpt": text[:500],
                "zero_shot_top_label": result["labels"][0],
                "zero_shot_top_score": float(result["scores"][0]),
                "zero_shot_all_labels": ";".join(result["labels"]),
                "zero_shot_all_scores": ";".join(
                    str(round(float(x), 4)) for x in result["scores"]
                )
            })

        zs_df = pd.DataFrame(zs_rows)

        zs_df.to_csv(
            OUTPUT_DIR / "zero_shot_theme_validation_sample.csv",
            index=False
        )

        zs_summary = (
            zs_df["zero_shot_top_label"]
            .value_counts()
            .reset_index()
        )

        zs_summary.columns = ["zero_shot_top_label", "count"]
        zs_summary["percentage"] = 100 * zs_summary["count"] / zs_summary["count"].sum()

        zs_summary.to_csv(
            OUTPUT_DIR / "zero_shot_theme_summary.csv",
            index=False
        )

        items = items.merge(
            zs_df[["id", "zero_shot_top_label", "zero_shot_top_score"]],
            on="id",
            how="left",
            suffixes=("", "_new")
        )

        items["zero_shot_top_label"] = items["zero_shot_top_label_new"].combine_first(
            items["zero_shot_top_label"]
        )

        items["zero_shot_top_score"] = items["zero_shot_top_score_new"].combine_first(
            items["zero_shot_top_score"]
        )

        items = items.drop(
            columns=["zero_shot_top_label_new", "zero_shot_top_score_new"],
            errors="ignore"
        )

    except Exception as e:
        warnings.warn(f"Zero-shot classification skipped because of error: {e}")


# ============================================================
# SAVE CLEAN DATA
# ============================================================

posts_out = posts.copy()
comments_out = comments.copy()

posts_out = posts_out.drop(columns=["author"], errors="ignore")
comments_out = comments_out.drop(columns=["author"], errors="ignore")

posts_out.to_csv(DATA_DIR / "kitchencels_posts_clean.csv", index=False, encoding="utf-8")
comments_out.to_csv(DATA_DIR / "kitchencels_comments_clean.csv", index=False, encoding="utf-8")

items_out_cols = [
    "id", "item_type", "subreddit", "author_anon", "created_dt",
    "created_date", "month", "text", "clean_text", "token_text",
    "score", "num_comments", "link_flair_text", "dominant_theme",
    "n_theme_flags", "vader_compound", "vader_sentiment_label",
    "roberta_sentiment_label", "roberta_sentiment_score",
    "zero_shot_top_label", "zero_shot_top_score", "permalink"
]

items_out_cols = items_out_cols + THEME_FLAG_COLS + THEME_COUNT_COLS

items[items_out_cols].to_csv(
    DATA_DIR / "kitchencels_items_clean.csv",
    index=False,
    encoding="utf-8"
)

sample = items[items_out_cols].sample(
    min(1000, len(items)),
    random_state=RANDOM_STATE
)

sample.to_csv(
    DATA_DIR / "representative_sample_for_submission.csv",
    index=False,
    encoding="utf-8"
)


# ============================================================
# DATASET SUMMARY
# ============================================================

summary = {
    "subreddit": SUBREDDIT_NAME,
    "n_posts": int(len(posts)),
    "n_comments": int(len(comments)),
    "n_total_items": int(len(items)),
    "n_unique_anonymised_authors": int(items["author_anon"].nunique()),
    "start_date": str(items["created_date"].min()),
    "end_date": str(items["created_date"].max()),
    "n_removed_or_unknown_authors": int((items["author_anon"] == "unknown").sum()),
    "n_flairs": int(posts["link_flair_text"].fillna("").nunique()),
    "roberta_sentiment_run": bool(USE_ROBERTA_SENTIMENT),
    "zero_shot_run": bool(USE_ZERO_SHOT),
    "zero_shot_sample_size": int(min(ZERO_SHOT_SAMPLE_SIZE, len(items))),
}

with open(OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nDataset summary:")
print(json.dumps(summary, indent=2))


# ============================================================
# INITIAL INSIGHTS VISUALISATIONS
# ============================================================

plt.figure(figsize=(7, 5))
items["item_type"].value_counts().plot(kind="bar")
plt.title("Dataset composition: posts and comments")
plt.xlabel("Item type")
plt.ylabel("Count")
plt.xticks(rotation=0)
save_fig("01_dataset_composition_posts_comments.png")

monthly = items.groupby(["month", "item_type"]).size().unstack(fill_value=0)

plt.figure(figsize=(10, 5))
monthly.plot(kind="line", marker="o")
plt.title("Monthly activity in r/kitchencels")
plt.xlabel("Month")
plt.ylabel("Number of items")
plt.xticks(rotation=45)
save_fig("02_monthly_activity.png")

flairs = posts["link_flair_text"].fillna("").replace("", "No flair").value_counts().head(12)

plt.figure(figsize=(9, 5))
flairs.sort_values().plot(kind="barh")
plt.title("Top post flairs")
plt.xlabel("Number of posts")
plt.ylabel("Flair")
save_fig("03_top_flairs.png")

plt.figure(figsize=(8, 5))
items["score"].clip(lower=-20, upper=200).hist(bins=40)
plt.title("Score distribution, clipped at -20 and 200")
plt.xlabel("Score")
plt.ylabel("Number of items")
save_fig("04_score_distribution.png")

author_counts = items.loc[items["author_anon"] != "unknown", "author_anon"].value_counts()
top_200 = author_counts.head(200)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(top_200) + 1), top_200.values)
plt.title("Distribution of anonymised user activity")
plt.xlabel("Anonymised user rank")
plt.ylabel("Number of posts/comments")
save_fig("05_author_activity_distribution.png")

theme_counts = items["dominant_theme"].value_counts().head(12)

plt.figure(figsize=(9, 5))
theme_counts.sort_values().plot(kind="barh")
plt.title("Dominant custom theme distribution")
plt.xlabel("Number of items")
plt.ylabel("Dominant theme")
save_fig("06_dominant_theme_distribution.png")

plt.figure(figsize=(9, 5))
plot_df = theme_summary_df.sort_values("items_with_theme")
plt.barh(plot_df["theme"], plot_df["items_with_theme"])
plt.title("Custom affective/theme lexicon categories")
plt.xlabel("Number of posts/comments")
plt.ylabel("Category")
save_fig("07_custom_affective_theme_categories.png")


# ============================================================
# SENTIMENT VISUALISATIONS
# ============================================================

if USE_VADER_BASELINE:
    plt.figure(figsize=(6, 6))
    items["vader_sentiment_label"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title("VADER baseline sentiment distribution")
    plt.ylabel("")
    save_fig("08_vader_sentiment_distribution.png")

if items["roberta_sentiment_label"].notna().sum() > 0:
    plt.figure(figsize=(6, 6))
    items["roberta_sentiment_label"].dropna().value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title("RoBERTa sentiment distribution")
    plt.ylabel("")
    save_fig("09_roberta_sentiment_distribution.png")

    roberta_theme = (
        items[items["roberta_sentiment_label"].notna()]
        .groupby(["dominant_theme", "roberta_sentiment_label"])
        .size()
        .unstack(fill_value=0)
    )

    top_theme_index = roberta_theme.sum(axis=1).sort_values(ascending=False).head(10).index
    roberta_theme = roberta_theme.loc[top_theme_index]

    roberta_theme.to_csv(OUTPUT_DIR / "roberta_sentiment_by_theme.csv")

    plt.figure(figsize=(10, 6))
    roberta_theme.plot(kind="bar")
    plt.title("RoBERTa sentiment by dominant custom theme")
    plt.xlabel("Dominant theme")
    plt.ylabel("Number of items")
    plt.xticks(rotation=45, ha="right")
    save_fig("10_roberta_sentiment_by_theme.png")

if USE_VADER_BASELINE:
    vader_theme = (
        items.groupby(["dominant_theme", "vader_sentiment_label"])
        .size()
        .unstack(fill_value=0)
    )

    top_theme_index = vader_theme.sum(axis=1).sort_values(ascending=False).head(10).index
    vader_theme = vader_theme.loc[top_theme_index]

    vader_theme.to_csv(OUTPUT_DIR / "vader_sentiment_by_theme.csv")

    plt.figure(figsize=(10, 6))
    vader_theme.plot(kind="bar")
    plt.title("VADER baseline sentiment by dominant custom theme")
    plt.xlabel("Dominant theme")
    plt.ylabel("Number of items")
    plt.xticks(rotation=45, ha="right")
    save_fig("11_vader_sentiment_by_theme.png")

monthly_roberta = (
    items[items["roberta_sentiment_label"].notna()]
    .groupby(["month", "roberta_sentiment_label"])
    .size()
    .unstack(fill_value=0)
)

if not monthly_roberta.empty:
    monthly_roberta.to_csv(OUTPUT_DIR / "monthly_roberta_sentiment_counts.csv")

    plt.figure(figsize=(10, 5))
    monthly_roberta.plot(kind="line", marker="o")
    plt.title("Monthly RoBERTa sentiment counts")
    plt.xlabel("Month")
    plt.ylabel("Number of items")
    plt.xticks(rotation=45)
    save_fig("12_monthly_roberta_sentiment_counts.png")


# ============================================================
# ZERO-SHOT VISUALISATION
# ============================================================

if items["zero_shot_top_label"].notna().sum() > 0:
    zs_counts = items["zero_shot_top_label"].dropna().value_counts()

    plt.figure(figsize=(10, 5))
    zs_counts.sort_values().plot(kind="barh")
    plt.title("Zero-shot theme classification sample")
    plt.xlabel("Number of sampled items")
    plt.ylabel("Zero-shot top label")
    save_fig("13_zero_shot_theme_sample.png")


# ============================================================
# TERM FREQUENCY
# ============================================================

term_rows = []

for item_type, group in items.groupby("item_type"):
    tokens = []
    for toks in group["tokens"]:
        tokens.extend(toks)

    for term, count in Counter(tokens).most_common(50):
        term_rows.append({
            "group": item_type,
            "term": term,
            "count": count
        })

for theme, group in items.groupby("dominant_theme"):
    tokens = []
    for toks in group["tokens"]:
        tokens.extend(toks)

    for term, count in Counter(tokens).most_common(30):
        term_rows.append({
            "group": f"theme_{theme}",
            "term": term,
            "count": count
        })

term_df = pd.DataFrame(term_rows)
term_df.to_csv(OUTPUT_DIR / "top_terms_by_group.csv", index=False)

for group_name in term_df["group"].unique():
    plot_df = term_df[term_df["group"] == group_name].head(15)
    if plot_df.empty:
        continue

    safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", group_name)

    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["term"][::-1], plot_df["count"][::-1])
    plt.title(f"Top terms: {group_name}")
    plt.xlabel("Frequency")
    plt.ylabel("Term")
    save_fig(f"14_top_terms_{safe_name}.png")


# ============================================================
# COMMUNITY SLANG ANALYSIS
# ============================================================

SLANG_TERMS = [
    "kitchencel", "kitchencels", "slop", "platemogging", "platemogged",
    "mog", "mogged", "fakecel", "truecel", "incel", "normie",
    "cope", "cooked", "billions", "must cook", "looksmaxx", "maxx",
    "chad", "stacy", "foid", "moid", "brutal", "never began",
    "it's over", "it is over"
]

slang_rows = []

for term in SLANG_TERMS:
    count = items["text"].apply(lambda x: count_phrase_matches(str(x), [term])).sum()
    slang_rows.append({
        "term": term,
        "count": int(count)
    })

slang_df = pd.DataFrame(slang_rows).sort_values("count", ascending=False)
slang_df.to_csv(OUTPUT_DIR / "community_slang_frequency.csv", index=False)

plt.figure(figsize=(8, 5))
plot_df = slang_df.head(15)
plt.barh(plot_df["term"][::-1], plot_df["count"][::-1])
plt.title("Community slang frequency")
plt.xlabel("Frequency")
plt.ylabel("Term")
save_fig("15_community_slang_frequency.png")


# ============================================================
# BURSTINESS ANALYSIS
# ============================================================

exploded = items[["created_date", "token_text"]].copy()
exploded["token"] = exploded["token_text"].str.split()
exploded = exploded.explode("token").dropna(subset=["token"])

word_day = (
    exploded.groupby(["token", "created_date"])
    .size()
    .reset_index(name="count")
)

burst = (
    word_day.groupby("token")
    .agg(
        total_count=("count", "sum"),
        max_daily_count=("count", "max"),
        mean_active_daily_count=("count", "mean"),
        active_days=("created_date", "nunique")
    )
    .reset_index()
)

burst = burst[burst["total_count"] >= 10].copy()

burst["burstiness_score"] = burst["max_daily_count"] / (
    burst["mean_active_daily_count"] + 1
)

burst = burst.sort_values("burstiness_score", ascending=False)
burst.to_csv(OUTPUT_DIR / "burstiness_terms.csv", index=False)

plt.figure(figsize=(8, 5))
plot_df = burst.head(15)
plt.barh(plot_df["token"][::-1], plot_df["burstiness_score"][::-1])
plt.title("Top bursty terms")
plt.xlabel("Burstiness score")
plt.ylabel("Term")
save_fig("16_burstiness_terms.png")


# ============================================================
# TOPIC MODELLING WITH SKLEARN LDA
# ============================================================

topic_docs = items.loc[items["token_text"].str.len() > 20, "token_text"].dropna()

if len(topic_docs) > MAX_LDA_DOCS:
    topic_docs = topic_docs.sample(MAX_LDA_DOCS, random_state=RANDOM_STATE)

if len(topic_docs) >= 50:
    min_df = 5 if len(topic_docs) >= 1000 else 2

    vectorizer = CountVectorizer(
        max_df=0.75,
        min_df=min_df,
        max_features=4000
    )

    X = vectorizer.fit_transform(topic_docs)

    k_values = range(2, 9)
    model_scores = []

    for k in k_values:
        lda = LatentDirichletAllocation(
            n_components=k,
            random_state=RANDOM_STATE,
            learning_method="batch",
            max_iter=20
        )

        lda.fit(X)

        model_scores.append({
            "n_topics": k,
            "log_likelihood": lda.score(X),
            "perplexity": lda.perplexity(X)
        })

    scores_df = pd.DataFrame(model_scores)
    scores_df.to_csv(OUTPUT_DIR / "lda_model_scores.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(scores_df["n_topics"], scores_df["perplexity"], marker="o")
    plt.title("LDA perplexity by number of topics")
    plt.xlabel("Number of topics")
    plt.ylabel("Perplexity")
    save_fig("17_lda_perplexity.png")

    plt.figure(figsize=(8, 5))
    plt.plot(scores_df["n_topics"], scores_df["log_likelihood"], marker="o")
    plt.title("LDA log-likelihood by number of topics")
    plt.xlabel("Number of topics")
    plt.ylabel("Log-likelihood")
    save_fig("18_lda_log_likelihood.png")

    best_k = int(scores_df.sort_values("perplexity").iloc[0]["n_topics"])

    lda_final = LatentDirichletAllocation(
        n_components=best_k,
        random_state=RANDOM_STATE,
        learning_method="batch",
        max_iter=30
    )

    lda_final.fit(X)
    feature_names = vectorizer.get_feature_names_out()

    topic_rows = []

    for topic_idx, topic in enumerate(lda_final.components_):
        top_indices = topic.argsort()[:-16:-1]

        for rank, i in enumerate(top_indices, start=1):
            topic_rows.append({
                "topic": topic_idx + 1,
                "rank": rank,
                "term": feature_names[i],
                "weight": float(topic[i])
            })

    topic_terms = pd.DataFrame(topic_rows)
    topic_terms.to_csv(OUTPUT_DIR / "lda_topic_terms.csv", index=False)

    for topic in sorted(topic_terms["topic"].unique()):
        plot_df = topic_terms[topic_terms["topic"] == topic].head(12)

        plt.figure(figsize=(8, 5))
        plt.barh(plot_df["term"][::-1], plot_df["weight"][::-1])
        plt.title(f"LDA topic {topic}: top terms")
        plt.xlabel("Term weight")
        plt.ylabel("Term")
        save_fig(f"19_lda_topic_{topic}_terms.png")

else:
    print("Not enough documents for topic modelling.")


# ============================================================
# NETWORK ANALYSIS: DIRECTED USER REPLY NETWORK
# ============================================================

post_author = dict(zip(posts["id"].astype(str), posts["author_anon"]))
comment_author = dict(zip(comments["id"].astype(str), comments["author_anon"]))

edge_counter = Counter()
edge_score_sum = defaultdict(float)
edge_examples = defaultdict(int)
self_replies = 0
unresolved_parent_author = 0

for row in tqdm(
    comments.itertuples(index=False),
    total=len(comments),
    desc="Building reply network"
):
    source = row.author_anon

    if source == "unknown":
        continue

    parent_id = str(row.parent_id)
    target = None

    if parent_id.startswith("t3_"):
        post_id = parent_id.replace("t3_", "", 1)
        target = post_author.get(post_id)

    elif parent_id.startswith("t1_"):
        comment_id = parent_id.replace("t1_", "", 1)
        target = comment_author.get(comment_id)

    if target is None or target == "unknown":
        unresolved_parent_author += 1
        continue

    if source == target:
        self_replies += 1
        continue

    edge_counter[(source, target)] += 1

    try:
        edge_score_sum[(source, target)] += float(row.score)
    except Exception:
        pass

    edge_examples[(source, target)] += 1


edge_rows = []

for (source, target), weight in edge_counter.items():
    edge_rows.append({
        "source": source,
        "target": target,
        "weight": int(weight),
        "distance": float(1 / weight),
        "avg_comment_score": edge_score_sum[(source, target)] / max(edge_examples[(source, target)], 1)
    })

edges_df = pd.DataFrame(edge_rows)
edges_df.to_csv(OUTPUT_DIR / "user_reply_network_edges.csv", index=False)

G = nx.DiGraph()

for row in edges_df.itertuples(index=False):
    G.add_edge(
        row.source,
        row.target,
        weight=int(row.weight),
        distance=float(row.distance),
        avg_comment_score=float(row.avg_comment_score)
    )

network_summary = {
    "nodes": int(G.number_of_nodes()),
    "edges": int(G.number_of_edges()),
    "density": float(nx.density(G)) if G.number_of_nodes() > 1 else 0,
    "reciprocity": float(nx.reciprocity(G)) if G.number_of_edges() > 0 and nx.reciprocity(G) is not None else None,
    "weakly_connected_components": int(nx.number_weakly_connected_components(G)) if G.number_of_nodes() > 0 else 0,
    "self_replies_excluded": int(self_replies),
    "unresolved_parent_author_comments": int(unresolved_parent_author),
}

with open(OUTPUT_DIR / "user_reply_network_summary.json", "w", encoding="utf-8") as f:
    json.dump(network_summary, f, indent=2)

print("\nNetwork summary:")
print(json.dumps(network_summary, indent=2))

metrics = pd.DataFrame({"node": list(G.nodes())})

if G.number_of_nodes() > 0:
    metrics["in_degree_weighted"] = metrics["node"].map(dict(G.in_degree(weight="weight"))).fillna(0)
    metrics["out_degree_weighted"] = metrics["node"].map(dict(G.out_degree(weight="weight"))).fillna(0)
    metrics["degree_centrality"] = metrics["node"].map(nx.degree_centrality(G)).fillna(0)

    if G.number_of_edges() > 0:
        metrics["pagerank"] = metrics["node"].map(nx.pagerank(G, weight="weight")).fillna(0)

        if G.number_of_nodes() > 500:
            bet = nx.betweenness_centrality(
                G,
                k=300,
                seed=RANDOM_STATE,
                weight="distance",
                normalized=True
            )
        else:
            bet = nx.betweenness_centrality(
                G,
                weight="distance",
                normalized=True
            )

        metrics["betweenness_centrality"] = metrics["node"].map(bet).fillna(0)
    else:
        metrics["pagerank"] = 0
        metrics["betweenness_centrality"] = 0

    UG = G.to_undirected()

    if UG.number_of_edges() > 0:
        communities = list(nx.community.greedy_modularity_communities(UG, weight="weight"))

        community_map = {}
        for i, community in enumerate(communities, start=1):
            for node in community:
                community_map[node] = i

        metrics["community"] = metrics["node"].map(community_map).fillna(0).astype(int)
    else:
        metrics["community"] = 0

metrics = metrics.sort_values("pagerank", ascending=False)

metrics.to_csv(OUTPUT_DIR / "user_reply_network_node_metrics.csv", index=False)

centrality_cols = [
    "in_degree_weighted",
    "out_degree_weighted",
    "degree_centrality",
    "pagerank",
    "betweenness_centrality"
]

if not metrics.empty:
    metrics[centrality_cols].describe().to_csv(
        OUTPUT_DIR / "user_reply_centrality_summary_statistics.csv"
    )

    metrics.head(25).to_csv(
        OUTPUT_DIR / "top_25_central_users.csv",
        index=False
    )

if G.number_of_nodes() > 2 and G.number_of_edges() > 1:
    top_nodes = metrics.head(min(150, len(metrics)))["node"].tolist()
    H = G.subgraph(top_nodes).copy()

    plt.figure(figsize=(12, 10))

    pos = nx.spring_layout(H, seed=RANDOM_STATE, weight="weight", k=0.8)

    metric_index = metrics.set_index("node")

    node_sizes = []
    node_colors = []

    for n in H.nodes():
        pr = float(metric_index.loc[n, "pagerank"]) if n in metric_index.index else 0
        community = int(metric_index.loc[n, "community"]) if n in metric_index.index else 0

        node_sizes.append(80 + 8000 * pr)
        node_colors.append(community)

    edge_widths = [
        0.4 + 0.25 * H[u][v].get("weight", 1)
        for u, v in H.edges()
    ]

    nx.draw_networkx_edges(
        H,
        pos,
        alpha=0.25,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=8,
        width=edge_widths
    )

    nx.draw_networkx_nodes(
        H,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        alpha=0.85
    )

    label_nodes = metrics.head(12)["node"].tolist()
    labels = {n: n for n in label_nodes if n in H.nodes()}

    nx.draw_networkx_labels(
        H,
        pos,
        labels=labels,
        font_size=8
    )

    plt.title("r/kitchencels directed user-reply network")
    plt.axis("off")
    save_fig("20_user_reply_network.png")

    nx.write_gexf(G, OUTPUT_DIR / "user_reply_network_for_gephi.gexf")


# ============================================================
# THEME CO-OCCURRENCE NETWORK
# ============================================================

theme_edges = Counter()

for row in items.itertuples(index=False):
    matched = []

    for theme in THEME_LEXICONS.keys():
        if getattr(row, f"{theme}_flag") == 1:
            matched.append(theme)

    matched = sorted(set(matched))

    if len(matched) < 2:
        continue

    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            theme_edges[(matched[i], matched[j])] += 1

theme_edge_rows = [
    {"source": s, "target": t, "weight": w}
    for (s, t), w in theme_edges.items()
]

theme_edges_df = pd.DataFrame(theme_edge_rows)
theme_edges_df.to_csv(OUTPUT_DIR / "theme_cooccurrence_network_edges.csv", index=False)

TG = nx.Graph()

for row in theme_edges_df.itertuples(index=False):
    TG.add_edge(row.source, row.target, weight=int(row.weight))

if TG.number_of_edges() > 0:
    theme_metrics = pd.DataFrame({"node": list(TG.nodes())})
    theme_metrics["weighted_degree"] = theme_metrics["node"].map(dict(TG.degree(weight="weight"))).fillna(0)
    theme_metrics["degree_centrality"] = theme_metrics["node"].map(nx.degree_centrality(TG)).fillna(0)
    theme_metrics["betweenness_centrality"] = theme_metrics["node"].map(
        nx.betweenness_centrality(TG, weight=None)
    ).fillna(0)

    theme_metrics = theme_metrics.sort_values("weighted_degree", ascending=False)
    theme_metrics.to_csv(OUTPUT_DIR / "theme_cooccurrence_node_metrics.csv", index=False)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(TG, seed=RANDOM_STATE, weight="weight")

    node_sizes = [
        200 + 15 * TG.degree(n, weight="weight")
        for n in TG.nodes()
    ]

    edge_widths = [
        0.5 + 0.03 * TG[u][v].get("weight", 1)
        for u, v in TG.edges()
    ]

    nx.draw_networkx_edges(TG, pos, width=edge_widths, alpha=0.35)
    nx.draw_networkx_nodes(TG, pos, node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_labels(TG, pos, font_size=8)

    plt.title("Theme co-occurrence network")
    plt.axis("off")
    save_fig("21_theme_cooccurrence_network.png")

    nx.write_gexf(TG, OUTPUT_DIR / "theme_cooccurrence_network_for_gephi.gexf")


# ============================================================
# SLANG CO-OCCURRENCE NETWORK
# ============================================================

slang_edges = Counter()

for text in items["text"].fillna("").astype(str):
    matched_terms = []

    for term in SLANG_TERMS:
        if count_phrase_matches(text, [term]) > 0:
            matched_terms.append(term)

    matched_terms = sorted(set(matched_terms))

    if len(matched_terms) < 2:
        continue

    for i in range(len(matched_terms)):
        for j in range(i + 1, len(matched_terms)):
            slang_edges[(matched_terms[i], matched_terms[j])] += 1

slang_edge_rows = [
    {"source": s, "target": t, "weight": w}
    for (s, t), w in slang_edges.items()
]

slang_edges_df = pd.DataFrame(slang_edge_rows)
slang_edges_df.to_csv(OUTPUT_DIR / "slang_cooccurrence_network_edges.csv", index=False)

SG = nx.Graph()

for row in slang_edges_df.itertuples(index=False):
    SG.add_edge(row.source, row.target, weight=int(row.weight))

if SG.number_of_edges() > 0:
    slang_metrics = pd.DataFrame({"node": list(SG.nodes())})
    slang_metrics["weighted_degree"] = slang_metrics["node"].map(dict(SG.degree(weight="weight"))).fillna(0)
    slang_metrics["degree_centrality"] = slang_metrics["node"].map(nx.degree_centrality(SG)).fillna(0)
    slang_metrics = slang_metrics.sort_values("weighted_degree", ascending=False)

    slang_metrics.to_csv(OUTPUT_DIR / "slang_cooccurrence_node_metrics.csv", index=False)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(SG, seed=RANDOM_STATE, weight="weight")

    node_sizes = [
        200 + 20 * SG.degree(n, weight="weight")
        for n in SG.nodes()
    ]

    edge_widths = [
        0.5 + 0.05 * SG[u][v].get("weight", 1)
        for u, v in SG.edges()
    ]

    nx.draw_networkx_edges(SG, pos, width=edge_widths, alpha=0.35)
    nx.draw_networkx_nodes(SG, pos, node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_labels(SG, pos, font_size=8)

    plt.title("Community slang co-occurrence network")
    plt.axis("off")
    save_fig("22_slang_cooccurrence_network.png")

    nx.write_gexf(SG, OUTPUT_DIR / "slang_cooccurrence_network_for_gephi.gexf")


# ============================================================
# ENGAGEMENT TABLES
# ============================================================

top_posts = posts.copy()
top_posts["title_clean"] = top_posts["title"].fillna("").astype(str)
top_posts["engagement_score"] = top_posts["score"].fillna(0) + top_posts["num_comments"].fillna(0)

top_posts_out = top_posts.sort_values("engagement_score", ascending=False).head(50)

top_posts_out[[
    "id", "author_anon", "created_dt", "title_clean", "score",
    "num_comments", "link_flair_text", "permalink", "engagement_score"
]].to_csv(OUTPUT_DIR / "top_50_posts_by_engagement.csv", index=False)

author_summary = (
    items[items["author_anon"] != "unknown"]
    .groupby("author_anon")
    .agg(
        n_items=("id", "count"),
        avg_score=("score", "mean"),
        total_score=("score", "sum"),
        first_seen=("created_dt", "min"),
        last_seen=("created_dt", "max")
    )
    .reset_index()
    .sort_values("n_items", ascending=False)
)

author_summary.to_csv(OUTPUT_DIR / "anonymised_author_activity_summary.csv", index=False)


# ============================================================
# MANUAL VALIDATION SAMPLE
# ============================================================

validation_cols = [
    "id", "item_type", "created_dt", "dominant_theme", "vader_sentiment_label",
    "roberta_sentiment_label", "zero_shot_top_label", "text"
]

manual_validation = items[validation_cols].sample(
    min(150, len(items)),
    random_state=RANDOM_STATE
).copy()

manual_validation["manual_theme"] = ""
manual_validation["manual_irony_present"] = ""
manual_validation["manual_distress_language_present"] = ""
manual_validation["notes"] = ""

manual_validation.to_csv(
    OUTPUT_DIR / "manual_validation_sample_to_label.csv",
    index=False,
    encoding="utf-8"
)


# ============================================================
# DONE
# ============================================================

print("\nAnalysis complete.")
print("Important output folders:")
print("  data/")
print("  outputs/")
print("  figures/")
print("\nZip them with:")
print("Compress-Archive -Path data,outputs,figures -DestinationPath kitchencels_analysis_results_v2.zip -Force")