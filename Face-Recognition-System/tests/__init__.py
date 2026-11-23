import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(_ROOT / ".keras")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(_ROOT / ".matplotlib")
