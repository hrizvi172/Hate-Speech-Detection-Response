import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR        = os.path.join(BASE_DIR, "outputs")

# ── Dataset settings ───────────────────────────────────
RANDOM_SEED   = 42
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15

# ── Label mapping ──────────────────────────────────────
# 0 = hate speech, 1 = offensive language, 2 = neither
LABEL_NAMES = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}

# ── Model settings ─────────────────────────────────────
BERT_MODEL_NAME     = "bert-base-uncased"
ROBERTA_MODEL_NAME  = "roberta-base"
MAX_TOKEN_LENGTH    = 128
BATCH_SIZE          = 16
EPOCHS              = 3
LEARNING_RATE       = 2e-5