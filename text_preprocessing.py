"""Text cleaning pipeline (same steps as in the notebook)."""
import string

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("english"))
except Exception:  # offline / nltk missing -> fall back to sklearn's list
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP_WORDS = set(ENGLISH_STOP_WORDS)

_PUNC_TABLE = str.maketrans("", "", string.punctuation)


def remove_punc(txt: str) -> str:
    return txt.translate(_PUNC_TABLE)


def remove_numbers(txt: str) -> str:
    return "".join(ch for ch in txt if not ch.isdigit())


def remove_emojis(txt: str) -> str:
    return "".join(ch for ch in txt if ch.isascii())


def remove_stopwords(txt: str) -> str:
    return " ".join(w for w in txt.split() if w not in STOP_WORDS)


def clean_text(txt: str) -> str:
    """lowercase -> strip punctuation -> strip numbers -> strip emojis -> strip stopwords"""
    txt = str(txt).lower()
    txt = remove_punc(txt)
    txt = remove_numbers(txt)
    txt = remove_emojis(txt)
    txt = remove_stopwords(txt)
    return txt
