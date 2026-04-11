"""Project entry point for the Modular RAG MCP Server scaffold."""

from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def main() -> None:
    _ensure_src_on_path()
    from core.settings import SettingsError, load_settings
    from observability.logger import get_logger

    logger = get_logger("bootstrap")
    settings_path = Path(__file__).resolve().parent / "config" / "settings.yaml"

    try:
        settings = load_settings(str(settings_path))
    except SettingsError as exc:
        print(f"Failed to load settings: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    logger.info("Settings loaded from %s", settings_path)
    logger.info(
        "Bootstrap ready: llm.provider=%s, embedding.provider=%s",
        settings.llm.get("provider"),
        settings.embedding.get("provider"),
    )
    print("Modular RAG MCP Server scaffold initialized.")


if __name__ == "__main__":
    main()
