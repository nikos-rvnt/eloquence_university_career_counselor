from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import Literal, Deque, Dict, Any, Tuple, List
from collections import deque
import normalization


class SlotExtraction(BaseModel):

    gender: str | None = None
    name: str | None = None
    region: str | None = None
    field_of_interest: str | None = None
    academic_background: str | None = None
    mobility_preference: str | None = None
    career_goal: str | None = None


@dataclass
class UserProfileState:

    profile: Dict[str, Any] = field(
        default_factory=lambda: {
            "gender": None,
            "name": None,
            "region": None,
            "field_of_interest": None,
            "academic_background": None,
            "mobility_preference": None,
            "career_goal": None,
        }
    )

    confidence: Dict[str, float] = field(
        default_factory=lambda: {
            "name": 0.0,
            "region": 0.0,
            "field_of_interest": 0.0,
            "academic_background": 0.0,
            "mobility_preference": 0.0,
            "career_goal": 0.0,
        }
    )

    sources: Dict[str, str | None] = field(
        default_factory=lambda: {
            "gender": None,
            "name": None,
            "region": None,
            "field_of_interest": None,
            "academic_background": None,
            "mobility_preference": None,
            "career_goal": None,
        }
    )

    def reset(self):

        for slot in self.profile:
            self.profile[slot] = None

        for slot in self.confidence:
            self.confidence[slot] = 0.0

        for slot in self.sources:
            self.sources[slot] = None

    def update_slot(self, slot: str, value: Any, confidence: float, source: str) -> bool:

        if value is None:
            return False

        current_conf = self.confidence.get(slot, 0.0)
        if self.profile.get(slot) is None or confidence >= current_conf:
            self.profile[slot] = value
            self.confidence[slot] = confidence
            self.sources[slot] = source

            return True

        return False


@dataclass
class DialogueTurn:
    speaker: Literal["user", "system"]
    text: str


class DialogueTurnsTracker:

    def __init__(self, max_turns: int = 50):
        self.turns: Deque[DialogueTurn] = deque(maxlen=max_turns)

    def add_user_turn(self, text: str) -> None:
        text = normalization._safe_strip(text)
        if text:
            self.turns.append(
                DialogueTurn(
                    speaker="user",
                    text=text,
                )
            )

    def add_system_turn(self, text: str):
        text = normalization._safe_strip(text)
        if text:
            self.turns.append(
                DialogueTurn(
                    speaker="system",
                    text=text,
                )
            )

    def reset(self):
        self.turns.clear()

    def get_recent_context(self, max_turns: int = 6) -> str:

        recent = list(self.turns)[-max_turns:]
        return "\n".join(f"{t.speaker}: {t.text}" for t in recent)

    def get_last_system_turn(self) -> str:

        for turn in reversed(self.turns):
            if turn.speaker == "system":
                return turn.text

        return ""


@dataclass(slots=True)
class DocumentChunk:
    """an indexed text chunk and the metadata associated with its university programme record"""

    text: str
    source: str
    chunk_id: int

    university: str | None = None
    course: str | None = None
    course_code: str | None = None
    degree: str | None = None
    type: str | None = None
    curriculum: str | None = None
    section: str | None = None
    url: str | None = None
    region: str | None = None
    explicit_region: str | None = None


class RecommendedDepartment(BaseModel):
    name: str
    why_it_matches: str
    source_files: list[str] = []
    url: str | None = None


class Recommendations(BaseModel):
    departments: list[RecommendedDepartment]
