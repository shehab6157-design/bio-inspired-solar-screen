import sys
from pathlib import Path

# make src/ importable from tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
