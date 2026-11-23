"""Who is talking, plus the synthetic data the tools read."""
import json
from dataclasses import dataclass
from pathlib import Path

MODEL = "openai:gpt-4o-mini"
FALLBACK_MODEL = "openai:gpt-4.1-mini"
DATA = Path(__file__).resolve().parent.parent / "data"
PATIENTS = json.loads((DATA / "patients.json").read_text())   # SYNTHETIC patients only


@dataclass
class CareContext:
    """Set by the app (login), never by the chat. Defaults make Studio and the chat UI work out of the box."""
    user_id: str = "sara"
    role: str = "patient"          # "patient" or "doctor"
    patient_token: str = "p-001"
    language: str = "English"
