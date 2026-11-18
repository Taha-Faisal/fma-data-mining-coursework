"""Run the task 1 audit from the project root.

AI-generated code (OpenAI Codex). Full user prompt:
"I want you to implement the first task of the data mining project."
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fma_project.audit import main


if __name__ == "__main__":
    raise SystemExit(main())
