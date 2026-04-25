"""
Core data structures for the solver engine.
Every solver returns a SolutionResult containing a list of SolutionStep objects.
This uniform structure makes adding new solvers trivial.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import base64, io, json
import plotly.graph_objects as go
import plotly.io as pio


@dataclass
class SolutionStep:
    step_number: int
    title: str                        # e.g. "Compute mean of each feature"
    calculation: str                  # LaTeX-style or plain string showing the formula used
    result: Any                       # The actual computed value (scalar, list, matrix)
    explanation: str                  # Why this step exists — plain English
    visual_b64: Optional[str] = None  # Base64 PNG if this step has a plot
    hint_1: str = ""
    hint_2: str = ""
    hint_3: str = ""                  # Most specific hint (near full reveal)

    def to_dict(self):
        return {
            "step_number": self.step_number,
            "title": self.title,
            "calculation": self.calculation,
            "result": self._serialize(self.result),
            "explanation": self.explanation,
            "visual_b64": self.visual_b64,
            "hints": [self.hint_1, self.hint_2, self.hint_3],
        }

    def _serialize(self, val):
        if hasattr(val, "tolist"):
            return val.tolist()
        if isinstance(val, (int, float, str, list, dict, bool)):
            return val
        return str(val)


@dataclass
class SolutionResult:
    problem_type: str                 # e.g. "Linear Regression"
    input_summary: str                # Short description of what was given
    steps: list[SolutionStep] = field(default_factory=list)
    final_answer: Any = None
    related_topics: list[str] = field(default_factory=list)
    interview_framing: str = ""       # How this appears in data science interviews

    def to_dict(self):
        return {
            "problem_type": self.problem_type,
            "input_summary": self.input_summary,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self._serialize(self.final_answer),
            "related_topics": self.related_topics,
            "interview_framing": self.interview_framing,
        }

    def _serialize(self, val):
        if hasattr(val, "tolist"):
            return val.tolist()
        return val


def fig_to_b64(fig) -> str:
    """Convert a plotly figure to a base64 PNG string."""
    img_bytes = pio.to_image(fig, format="png", width=700, height=400)
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return encoded
