from pathlib import Path

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True) if p.suffix else p.mkdir(parents=True, exist_ok=True)