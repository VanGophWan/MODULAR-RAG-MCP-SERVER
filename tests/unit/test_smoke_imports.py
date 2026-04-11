from pathlib import Path
import sys


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def test_smoke_imports() -> None:
    _add_src_to_path()

    import mcp_server  # noqa: F401
    import core  # noqa: F401
    import ingestion  # noqa: F401
    import libs  # noqa: F401
    import observability  # noqa: F401
