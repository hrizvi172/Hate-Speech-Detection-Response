"""
preprocessor.py
---------------
Cleans and normalizes raw social media text.
Every other module imports clean_text() from here.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── One-time setup ─────────────────────────────────────────────────────────────
STOP_WORDS  = set(stopwords.words('english'))
LEMMATIZER  = WordNetLemmatizer()

# Words that LOOK like stopwords but carry meaning for hate speech detection
# e.g. "not", "no", "against" change the meaning completely — keep them
KEEP_WORDS  = {"not", "no", "nor", "against", "without", "never",
               "none", "nobody", "nothing", "nowhere"}
STOP_WORDS  = STOP_WORDS - KEEP_WORDS

# Social-media-specific noise words (no semantic value)
EXTRA_NOISE = {"rt", "via", "amp", "gt", "lt"}   # rt = retweet


def clean_text(text: str, 
               remove_stopwords: bool = True,
               lemmatize: bool = True) -> str:
    """
    Full cleaning pipeline for a single tweet.

    Steps
    -----
    1. Lowercase
    2. Remove URLs
    3. Remove @mentions
    4. Remove hashtag symbol  (keep the word after #)
    5. Remove HTML entities   (&amp; &#8216; etc.)
    6. Convert emojis → text  (basic mapping)
    7. Remove numbers
    8. Remove punctuation / special characters
    9. Tokenize
    10. Remove stopwords  (optional)
    11. Lemmatize         (optional)
    12. Remove very short tokens  (single letters)
    13. Re-join into a string
    """

    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs  (http/https/www patterns)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 3. Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # 4. Remove # symbol but keep the word  (#BlackLivesMatter → blacklivesmatter)
    text = re.sub(r'#', '', text)

    # 5. Remove HTML entities  (&amp; → ''  &#1234; → '')
    text = re.sub(r'&[a-z]+;|&#\d+;', '', text)

    # 6. Basic emoji → text  (the most common ones in hate speech datasets)
    EMOJI_MAP = {
        ':)'  : ' happy ',    ':('  : ' sad ',
        ':D'  : ' happy ',    ":'(" : ' sad ',
        ';)'  : ' wink ',     ':P'  : ' playful ',
        ':/'  : ' skeptical ', '>:(' : ' angry ',
    }
    for emoji, word in EMOJI_MAP.items():
        text = text.replace(emoji, word)

    # 7. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 8. Remove punctuation and special characters
    #    Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # 9. Tokenize  (split into list of words)
    tokens = word_tokenize(text)

    # 10. Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS
                                    and t not in EXTRA_NOISE]

    # 11. Lemmatize  (running → run, better → good)
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    # 12. Remove very short tokens  (single letters like 'a', 'i' after cleaning)
    tokens = [t for t in tokens if len(t) > 1]

    # 13. Rejoin
    return ' '.join(tokens)


def preprocess_dataframe(df, text_col: str = 'text') -> None:
    """
    Adds a 'clean_text' column to the dataframe in-place.
    Also adds 'token_count' for analysis.
    """
    print(f"Preprocessing {len(df)} samples...")
    df['clean_text']   = df[text_col].apply(clean_text)
    df['token_count']  = df['clean_text'].apply(lambda x: len(x.split()))

    # How many became empty after cleaning?
    empty = (df['clean_text'].str.strip() == '').sum()
    print(f"  Empty after cleaning : {empty} samples")
    print(f"  Avg tokens per tweet : {df['token_count'].mean():.1f}")
    print("Done.")