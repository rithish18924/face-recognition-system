from __future__ import annotations

import os
from pathlib import Path

# Ensure KERAS_HOME and MPLCONFIGDIR point inside the project directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(_PROJECT_ROOT / ".keras")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(_PROJECT_ROOT / ".matplotlib")

__version__ = "1.0.0"
