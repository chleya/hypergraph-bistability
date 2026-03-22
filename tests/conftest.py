import os
import tempfile
from pathlib import Path


_TEST_TEMP_DIR = Path(__file__).resolve().parent / ".tmp"
_TEST_TEMP_DIR.mkdir(exist_ok=True)

os.environ.setdefault("TMP", str(_TEST_TEMP_DIR))
os.environ.setdefault("TEMP", str(_TEST_TEMP_DIR))
tempfile.tempdir = str(_TEST_TEMP_DIR)
