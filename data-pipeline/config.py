"""Central configuration for the Bangla data collection pipeline."""

from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
STATE_DIR = DATA_DIR / "state"
REPORTS_DIR = DATA_DIR / "reports"
SEEDS_DIR = BASE_DIR / "seeds"

# Per-source raw directories
RAW_WIKIPEDIA = RAW_DIR / "wikipedia"
RAW_WIKISOURCE = RAW_DIR / "wikisource"
RAW_NEWSPAPERS = RAW_DIR / "newspapers"
RAW_LITERATURE = RAW_DIR / "literature"
RAW_WEB_CRAWL = RAW_DIR / "web_crawl"
RAW_BANGLISH = RAW_DIR / "banglish"
RAW_HF_CORPUS = RAW_DIR / "hf_corpus"

# --- Rate Limits ---
NEWSPAPER_RPS = 1.0          # requests per second per domain
WEB_CRAWL_RPS = 2.0          # requests per second
ARCHIVE_ORG_RPS = 1.0        # Internet Archive rate limit
REDDIT_DELAY = 2.0           # seconds between Reddit API calls

# --- Collection ---
WEB_CRAWL_MAX_DEPTH = 3
WEB_CRAWL_MAX_PAGES = 5000   # per seed domain
REQUEST_TIMEOUT = 30          # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0           # exponential backoff multiplier

USER_AGENT = (
    "BanglaLLM-DataCollector/1.0 "
    "(Research project; collecting public Bangla text; "
    "respects robots.txt and rate limits)"
)

# --- Processing Thresholds ---
LANG_DETECT_THRESHOLD = 0.65          # minimum fasttext confidence for 'bn'
CONTAMINATION_THRESHOLD = 0.2         # max allowed Hindi/Assamese secondary score
MIN_DOC_LENGTH = 100                  # characters
MIN_BENGALI_CHAR_RATIO = 0.50         # minimum Bengali script ratio
MAX_REPEAT_LINE_RATIO = 0.30          # maximum duplicate line ratio
WIKI_STUB_THRESHOLD = 200             # filter Wikipedia stubs below this

# --- Deduplication ---
MINHASH_NUM_PERM = 128                # MinHash permutations
MINHASH_SHINGLE_SIZE = 5              # character n-gram size
MINHASH_THRESHOLD = 0.8               # Jaccard similarity threshold for near-dedup

# --- Export ---
EXPORT_CHUNK_SIZE_MB = 100            # gzipped JSONL chunk size
EXPORT_COMPRESSION_LEVEL = 6          # gzip level (1-9)

# --- Unicode Ranges (for contamination detection) ---
DEVANAGARI_RANGE = (0x0900, 0x097F)                 # Hindi/Marathi/Sanskrit
BENGALI_RANGE = (0x0980, 0x09FF)                     # Bengali + Assamese overlap
ASSAMESE_ONLY_CHARS = {'\u09F0', '\u09F1'}           # ৰ, ৱ
ODIA_RANGE = (0x0B00, 0x0B7F)
TAMIL_RANGE = (0x0B80, 0x0BFF)
TELUGU_RANGE = (0x0C00, 0x0C7F)
KANNADA_RANGE = (0x0C80, 0x0CFF)
MALAYALAM_RANGE = (0x0D00, 0x0D7F)
GURMUKHI_RANGE = (0x0A00, 0x0A7F)
GUJARATI_RANGE = (0x0A80, 0x0AFF)

BLOCKED_SCRIPT_RANGES = [
    DEVANAGARI_RANGE,
    ODIA_RANGE,
    TAMIL_RANGE,
    TELUGU_RANGE,
    KANNADA_RANGE,
    MALAYALAM_RANGE,
    GURMUKHI_RANGE,
    GUJARATI_RANGE,
]

# --- HuggingFace Datasets ---
HF_DATASETS = {
    "cc100": {
        "path": "statmt/cc100",
        "name": "bn",
        "split": "train",
        "streaming": True,
    },
    "culturax": {
        "path": "uonlp/CulturaX",
        "name": "bn",
        "split": "train",
        "streaming": True,
    },
    "sangraha": {
        "path": "ai4bharat/sangraha",
        "name": "ben",
        "split": "train",
        "streaming": True,
    },
    # OSCAR - gated/suspended as of 2026. Enable if access is granted.
    # "oscar": {
    #     "path": "oscar-corpus/OSCAR-2301",
    #     "name": "bn",
    #     "split": "train",
    #     "streaming": True,
    # },
}

# --- Newspapers ---
NEWSPAPERS = {
    "prothomalo": {
        "base_url": "https://www.prothomalo.com",
        "sitemap_url": "https://www.prothomalo.com/sitemap.xml",
        "rps": NEWSPAPER_RPS,
    },
    "kalerkantho": {
        "base_url": "https://www.kalerkantho.com",
        "sitemap_url": "https://www.kalerkantho.com/sitemap.xml",
        "rps": NEWSPAPER_RPS,
    },
    "ittefaq": {
        "base_url": "https://www.ittefaq.com.bd",
        "sitemap_url": "https://www.ittefaq.com.bd/sitemap.xml",
        "rps": NEWSPAPER_RPS,
    },
}

# Excluded: Daily Star Bangla (ai-train=no in robots.txt)

# --- Google Drive ---
DRIVE_RAW_PATH = "My Drive/bangla-llm/raw/"
DRIVE_FINAL_PATH = "My Drive/bangla-llm/final/"


def ensure_dirs():
    """Create all data directories if they don't exist."""
    for d in [
        RAW_DIR, PROCESSED_DIR, FINAL_DIR, STATE_DIR, REPORTS_DIR,
        RAW_WIKIPEDIA, RAW_WIKISOURCE, RAW_NEWSPAPERS,
        RAW_LITERATURE, RAW_WEB_CRAWL, RAW_BANGLISH, RAW_HF_CORPUS,
    ]:
        d.mkdir(parents=True, exist_ok=True)
