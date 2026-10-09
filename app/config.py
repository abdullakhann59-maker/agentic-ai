"""All settings in one place. Values come from environment variables or the .env file."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # --- LLM (Ollama) ---
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    embed_model: str = "nomic-embed-text"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 8192

    # --- RAG thresholds (cosine similarity, 0..1). Tune these with your own data. ---
    chat_threshold: float = 0.80   # above this, a stored chat example is "the same question"
    task_threshold: float = 0.62   # above this, a stored task playbook is used as the plan
    doc_threshold: float = 0.55    # above this, a document chunk is added as context
    memory_threshold: float = 0.60
    tool_group_top_k: int = 3      # direct path: how many tool groups the agent gets

    # --- Agent limits ---
    max_steps: int = 12            # max LLM->tool rounds per command
    max_seconds: int = 300
    max_consecutive_errors: int = 3
    max_repeat_calls: int = 3      # same tool + same args this many times = stuck
    history_messages: int = 6      # previous chat turns sent to the LLM
    tool_output_chars: int = 3000

    # --- Storage ---
    data_dir: Path = BASE_DIR / "data"
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    sqlite_path: Path = BASE_DIR / "data" / "nexus.db"
    workspace_dir: Path = BASE_DIR / "data" / "workspace"

    # --- Browser / scraping ---
    allowed_domains: str = "localhost,127.0.0.1"   # comma separated; "*" allows all
    chrome_binary: str = ""          # leave empty to use the installed Chrome
    chromedriver_path: str = ""      # leave empty to let Selenium Manager find the driver
    chromedriver_args: str = ""      # e.g. "--disable-build-check"
    headless: bool = True
    scrape_delay_seconds: float = 1.0
    scrape_max_pages: int = 5
    user_agent: str = "NexusAgent/1.0 (portfolio project; polite scraper)"

    # --- Email ---
    email_demo_mode: bool = True     # True = log instead of sending
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""

    # --- API ---
    api_url: str = "http://localhost:8000"   # used by the Streamlit UI

    @property
    def allowed_domain_list(self) -> list[str]:
        return [d.strip().lower() for d in self.allowed_domains.split(",") if d.strip()]

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.chroma_dir, self.workspace_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
