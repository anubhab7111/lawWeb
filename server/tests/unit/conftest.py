import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://lawweb:lawweb@127.0.0.1:5432/lawweb_scratch"
)
os.environ.setdefault("JWT_SECRET", "test-secret")
