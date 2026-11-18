"""Run the task 2 manifest builder from the project root.

AI-generated code (OpenAI Codex). Full user prompt:
"lets move on to task 2"
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fma_project.manifest import main


if __name__ == "__main__":
    raise SystemExit(main())
