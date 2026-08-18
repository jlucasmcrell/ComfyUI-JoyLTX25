"""Auto-install dependencies for ComfyUI_JoyAI_Echo."""

import subprocess
import sys
from pathlib import Path

requirements = Path(__file__).parent / "requirements.txt"

if requirements.exists():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
    )
