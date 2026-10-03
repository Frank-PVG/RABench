from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]

def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path
