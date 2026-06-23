import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT_DEFAULT / ".env")

# Paths — all configurable via .env, with sensible defaults
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", _PROJECT_ROOT_DEFAULT))
ALL_DOCUMENTS_DIR = Path(os.environ.get("ALL_DOCUMENTS_DIR", PROJECT_ROOT / "all_documents"))
CONFLUENCE_DIR = Path(os.environ.get("CONFLUENCE_DIR", ALL_DOCUMENTS_DIR / "confluence"))
FIREFLIES_DIR = Path(os.environ.get("FIREFLIES_DIR", ALL_DOCUMENTS_DIR / "fireflies"))
GITHUB_DIR = Path(os.environ.get("GITHUB_DIR", ALL_DOCUMENTS_DIR / "github"))
GMAIL_DIR = Path(os.environ.get("GMAIL_DIR", ALL_DOCUMENTS_DIR / "gmail"))
GOOGLE_DRIVE_DIR = Path(os.environ.get("GOOGLE_DRIVE_DIR", ALL_DOCUMENTS_DIR / "google_drive"))
HUBSPOT_DIR = Path(os.environ.get("HUBSPOT_DIR", ALL_DOCUMENTS_DIR / "hubspot"))
LINEAR_DIR = Path(os.environ.get("LINEAR_DIR", ALL_DOCUMENTS_DIR / "linear"))
SLACK_DIR = Path(os.environ.get("SLACK_DIR", ALL_DOCUMENTS_DIR / "slack"))
JIRA_DIR = Path(os.environ.get("JIRA_DIR", ALL_DOCUMENTS_DIR / "jira"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "chunking_and_embedding" / "output"))

# Model
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-base-en")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "8192"))
# Documents exceeding this token count will use the segmented fallback
LONG_DOC_THRESHOLD = int(os.environ.get("LONG_DOC_THRESHOLD", "7500"))

# Chunking
MIN_CHUNK_TOKENS = int(os.environ.get("MIN_CHUNK_TOKENS", "50"))
MAX_CHUNK_TOKENS = int(os.environ.get("MAX_CHUNK_TOKENS", "800"))
MERGE_THRESHOLD_TOKENS = int(os.environ.get("MERGE_THRESHOLD_TOKENS", "100"))
OVERLAP_SENTENCES = int(os.environ.get("OVERLAP_SENTENCES", "2"))

# Qdrant — map .env names (CLUSTER_ENDPOINT/QDRANT_KEY) to standard names
if os.environ.get("CLUSTER_ENDPOINT") and not os.environ.get("QDRANT_URL"):
    os.environ["QDRANT_URL"] = os.environ["CLUSTER_ENDPOINT"].strip()
if os.environ.get("QDRANT_KEY") and not os.environ.get("QDRANT_API_KEY"):
    os.environ["QDRANT_API_KEY"] = os.environ["QDRANT_KEY"].strip()

QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "production_rag")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# Output
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
