import hashlib
import json
import glob
import re
import os
import numpy as np
import pickle
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Deque, Dict, Any, Tuple, List
from json_repair import repair_json
import sdialog
from sdialog.agents import Agent
from sdialog.personas import Persona
from sdialog.server import Server
from sdialog.orchestrators import (
    BaseOrchestrator,
    BasePersistentOrchestrator,
    LengthOrchestrator,
    SimpleReflexOrchestrator,
)
from sdialog.evaluation import LLMJudgeYesNo
from functools import lru_cache
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import difflib
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# Nofrom gliner2 import GLiNER2
from gliner import GLiNER

from langchain_ollama import ChatOllama

import config

if not os.getenv("OLLAMA_BASE_URL"):
    os.environ["OLLAMA_BASE_URL"] = "http://ollama:11435"
if not os.getenv("OLLAMA_HOST"):
    os.environ["OLLAMA_HOST"] = "http://ollama:11435"


def configure_application():
    sdialog.config.llm(config.MODEL_URI)
    sdialog.config.cache(True)

    os.makedirs(config.INDEX_CACHE_DIR, exist_ok=True)


@lru_cache(maxsize=2)
def get_llm(**kwargs) -> ChatOllama:
    return ChatOllama(model=config.MODEL_URI, temperature=0, **kwargs)


# schema
REQUIRED_SLOTS = {
    "gender": {
        "description": "The gender of the student.",
        "required": False,
    },
    "name": {
        "description": "First name of the student.",
        "required": True,
    },
    "region": {
        "description": "Region of Italy where the student lives (name of the city, hometown, suburb, etc.).",
        "required": True,
    },
    "field_of_interest": {
        "description": "Desired university study area or discipline.",
        "required": True,
    },
    "academic_background": {
        "description": "Current educational level and academic background.",
        "required": True,
    },
    "mobility_preference": {
        "description": "Whether the student can relocate or not and where would like to study.",
        "required": True,
    },
    "career_goal": {
        "description": "Career goals of the user after graduation.",
        "required": True,
    },
}

ENTITY_DESCRIPTIONS = {
    "NAME": "The first name of the user, if mentioned.",
    "REGION": "Macro-region of Italy where the user lives, such as north, center, south, islands.",
    "FIELD_OF_INTEREST": "University study area or academic interest such as computer science, psychology, arts, environmental science.",
    "CAREER_GOAL": "Career plans after graduation such as research, industry, clinical practice, education, consulting, or public sector.",
    "ACADEMIC_BACKGROUND": "Current or previous educational background such as high school, scientific high school, bachelor's degree, master's degree, engineering degree.",
    "MOBILITY_PREFERENCE": "Preference regarding the willingness of the user to relocate or not for university studies.",
}
# ENTITY_LABELS = list(ENTITY_DESCRIPTIONS.keys())

SLOT_ORDER = ["name", "academic_background", "field_of_interest", "career_goal", "region", "mobility_preference"]


class SlotExtraction(BaseModel):

    gender: str | None = None
    name: str | None = None
    region: str | None = None
    field_of_interest: str | None = None
    academic_background: str | None = None
    mobility_preference: str | None = None
    career_goal: str | None = None


slot_parser = PydanticOutputParser(pydantic_object=SlotExtraction)
# format_instructions = slot_parser.get_format_instructions()


def parse_slot_output(llm_output: str) -> SlotExtraction:

    try:
        return slot_parser.parse(llm_output)

    except Exception:
        try:
            repaired = repair_json(llm_output)
            return SlotExtraction.model_validate_json(repaired)

        except Exception:
            return SlotExtraction()


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


CONFIRMATION_THRESHOLD = 0.6

CITY_TO_REGION = {
    # North
    "milan": "north",
    "milano": "north",
    "turino": "north",
    "torino": "north",
    "bologna": "north",
    "venice": "north",
    "venezia": "north",
    "genoa": "north",
    "genova": "north",
    "trento": "north",
    "trieste": "north",
    "udine": "north",
    "verona": "north",
    "padova": "north",
    "modena": "north",
    "parma": "north",
    "ferrara": "north",
    "ravenna": "north",
    "rimini": "north",
    "forli": "north",
    "cesena": "north",
    "piacenza": "north",
    "cremona": "north",
    "mantova": "north",
    "bergamo": "north",
    "brescia": "north",
    "como": "north",
    "varese": "north",
    "sondrio": "north",
    "lecco": "north",
    "asti": "north",
    "alessandria": "north",
    "cuneo": "north",
    "novara": "north",
    "verbania": "north",
    "biella": "north",
    "ivrea": "north",
    "domodossola": "north",
    # Center
    "rome": "center",
    "roma": "center",
    "florence": "center",
    "firenze": "center",
    "pisa": "center",
    "perugia": "center",
    "ancona": "center",
    "l'aquila": "center",
    "teramo": "center",
    "pescara": "center",
    "chieti": "center",
    "siena": "center",
    "arezzo": "center",
    "livorno": "center",
    "lucca": "center",
    "grosseto": "center",
    "macerata": "center",
    "urbino": "center",
    "pesaro": "center",
    "fermo": "center",
    "viterbo": "center",
    "rieti": "center",
    "frosinone": "center",
    "latina": "center",
    "terni": "center",
    "orvieto": "center",
    "spoleto": "center",
    "civitanova": "center",
    # South
    "naples": "south",
    "napoli": "south",
    "en napoli": "south",
    "bari": "south",
    "salerno": "south",
    "foggia": "south",
    "lecce": "south",
    "brindisi": "south",
    "taranto": "south",
    "potenza": "south",
    "matera": "south",
    "catanzaro": "south",
    "crotone": "south",
    "cosenza": "south",
    "reggio calabria": "south",
    "vibo valentia": "south",
    "campobasso": "south",
    "isernia": "south",
    "avellino": "south",
    "benevento": "south",
    "caserta": "south",
    "cezze": "south",
    # Islands
    "palermo": "islands",
    "catania": "islands",
    "cagliari": "islands",
    "syracuse": "islands",
    "siracusa": "islands",
    "messina": "islands",
    "trapani": "islands",
    "agrigento": "islands",
    "ragusa": "islands",
    "sassari": "islands",
    "nuoro": "islands",
    "oristano": "islands",
}


def get_allowed_study_regions(
    home_region: str | None,
    mobility_preference: str | None,
) -> set[str] | None:
    """
    Convert the student's HOME REGION + MOBILITY PREFERENCE
    into the geographic constraint that should be applied to
    university retrieval.
    Returns:
        set[str]:
            Explicitly allowed study regions.
        None:
            No geographic restriction should be applied.
    Policy:
        strict_local -> only home region
        nearby_ok    -> home region + nearby macro-regions
        italy_ok     -> all Italy
        unknown/None -> no HARD restriction
    """

    if mobility_preference == "strict_local":
        if home_region in config.VALID_REGIONS:
            return {home_region}

        # a local constraint cannot be enforced if the home region itself is unknown.
        return None

    if mobility_preference == "nearby_ok":
        if home_region in config.NEARBY_REGIONS:
            return config.NEARBY_REGIONS[home_region]

        return None

    if mobility_preference == "italy_ok":
        return config.VALID_REGIONS.copy()

    return None


BACKGROUND_HINTS = {
    # High School Equivalents
    "scientific high school": "scientific high school",
    "liceo scientifico": "scientific high school",
    "high-school": "high school",
    "high school": "high school",
    "secondary school": "high school",
    "scuola superiore": "high school",
    "classical high school": "classical high school",
    "liceo classico": "classical high school",
    "humanities high school": "humanities high school",
    "liceo delle scienze umane": "humanities high school",
    "linguistic high school": "linguistic high school",
    "liceo linguistico": "linguistic high school",
    "art high school": "art high school",
    "liceo artistico": "art high school",
    "music high school": "music high school",
    "liceo musicale": "music high school",
    "technical institute": "technical institute",
    "istituto tecnico": "technical institute",
    "professional institute": "professional institute",
    "istituto professionale": "professional institute",
    # Degrees
    "bachelor": "bachelor's degree",
    "BSc": "bachelor's degree",
    "laurea triennale": "bachelor's degree",
    "master": "master's degree",
    "MSc": "master's degree",
    "laurea magistrale": "master's degree",
    "phd": "phd",
    "doctorate": "phd",
    "dottorato": "phd",
    "engineering degree": "engineering degree",
}

FIELD_HINTS = {
    # Computer Science & Technology
    "computer science": "computer science",
    "cs": "computer science",
    "informatics": "computer science",
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "data science": "data science",
    "machine learning": "data science",
    "software engineering": "software engineering",
    "programming": "computer science",
    "cybersecurity": "computer science",
    "information technology": "computer science",
    "computer engineering": "computer engineering",
    "civil engineering": "engineering",
    "mechanical engineering": "engineering",
    "electronic engineering": "engineering",
    "industrial engineering": "engineering",
    "biomedical engineering": "engineering",
    "materials engineering": "engineering",
    "telecommunications": "engineering",
    # Science
    "physics": "physics",
    "mathematics": "mathematics",
    "chemistry": "chemistry",
    "biology": "biology",
    "biotechnology": "biotechnology",
    "environmental science": "environmental science",
    "geology": "geology",
    "astronomy": "physics",
    "statistics": "mathematics",
    # Medicine & Health
    "medicine": "medicine",
    "surgery": "medicine",
    "pharmacy": "pharmacy",
    "nursing": "nursing",
    "physiotherapy": "physiotherapy",
    "dentistry": "dentistry",
    "veterinary": "veterinary",
    "nutrition": "nutrition",
    "psychology": "psychology",
    # Humanities & Social Sciences
    "psychology": "psychology",
    "social work": "social work",
    "sociology": "sociology",
    "education": "education",
    "pedagogy": "education",
    "philosophy": "philosophy",
    "history": "history",
    "literature": "literature",
    "linguistics": "linguistics",
    "law": "law",
    "economics": "economics",
    "management": "management",
    "marketing": "management",
    "political science": "political science",
    "international relations": "political science",
    "architecture": "architecture",
    "design": "design",
    "urban planning": "architecture",
    # Arts
    "arts": "arts",
    "fine arts": "arts",
    "music": "arts",
    "cinema": "arts",
    "theatre": "arts",
    "cultural heritage": "arts",
    "museology": "arts",
    "conservation": "arts",
    "visual arts": "arts",
    "graphic design": "design",
    "fashion design": "design",
}

CAREER_HINTS = {
    "research": "research",
    "academic": "research",
    "phd": "research",
    "scientist": "research",
    "researcher": "research",
    "industry": "industry",
    "business": "industry",
    "company": "industry",
    "corporate": "industry",
    "software engineer": "software engineering",
    "developer": "software engineering",
    "doctor": "clinical practice",
    "medical": "clinical practice",
    "physician": "clinical practice",
    "psychologist": "psychology practice",
    "therapist": "psychology practice",
    "teacher": "education",
    "professor": "education",
    "school": "education",
    "teaching": "education",
    "consulting": "consulting",
    "consultant": "consulting",
    "advisor": "consulting",
    "public sector": "public sector",
    "civil service": "public sector",
    "government": "public sector",
    "administration": "public sector",
    "politics": "public sector",
    "lawyer": "law",
    "solicitor": "law",
    "barrister": "law",
    "judge": "law",
    "engineer": "industry",
    "architect": "industry",
    "designer": "industry",
    "entrepreneur": "industry",
    "startup": "industry",
    "management": "industry",
    "healthcare": "clinical practice",
    "hospital": "clinical practice",
}


@dataclass
class DialogueTurn:
    speaker: Literal["user", "system"]
    text: str


class DialogueTurnsTracker:

    def __init__(self, max_turns: int = 50):
        self.turns: Deque[DialogueTurn] = deque(maxlen=max_turns)

    def add_user_turn(self, text: str) -> None:
        text = _safe_strip(text)
        if text:
            self.turns.append(
                DialogueTurn(
                    speaker="user",
                    text=text,
                )
            )

    def add_system_turn(self, text: str):
        text = _safe_strip(text)
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


def _safe_strip(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _fuzzy_match(value: str, choices: dict, cutoff: float = 0.85) -> str | None:
    """Return canonical value if any key is a close substring match."""
    value_lower = value.lower()
    # exact substring first
    for key, canonical in choices.items():
        if key in value_lower:
            return canonical
    # fuzzy ratio as fallback
    best_ratio = 0.0
    best_canon = None
    for key, canonical in choices.items():
        ratio = difflib.SequenceMatcher(None, value_lower, key).ratio()
        if ratio > best_ratio and ratio >= cutoff:
            best_ratio = ratio
            best_canon = canonical
    return best_canon


def normalize_region(value: Any) -> str | None:
    value = _safe_strip(value).lower()
    if not value:
        return None
    # direct keywords
    if any(w in value for w in ["north", "northern", "nord"]):
        return "north"
    if any(w in value for w in ["center", "central", "centro", "middle italy"]):
        return "center"
    if any(w in value for w in ["south", "southern", "sud"]):
        return "south"
    if any(w in value for w in ["island", "islands", "sicily", "sardinia", "sicilia", "sardegna"]):
        return "islands"
    # city lookup (unchanged)
    for city, region in CITY_TO_REGION.items():
        if city in value:
            return region
    return None


# For academic_background, field_of_interest, career_goal, mobility,
# replace the simple loop with _fuzzy_match using the existing hints:
def normalize_academic_background(value: Any) -> str | None:
    value = _safe_strip(value).lower()
    if not value:
        return None
    return _fuzzy_match(value, BACKGROUND_HINTS, cutoff=0.85) or value  # fallback to raw


def normalize_field_of_interest(value: Any) -> str | None:
    value = _safe_strip(value).lower()
    if not value:
        return None
    return _fuzzy_match(value, FIELD_HINTS, cutoff=0.85) or value


def normalize_career_goal(value: Any) -> str | None:
    value = _safe_strip(value).lower()
    if not value:
        return None
    return _fuzzy_match(value, CAREER_HINTS, cutoff=0.85) or value


def normalize_mobility(value: Any) -> str | None:
    value = _safe_strip(value).lower()
    if not value:
        return None
    # keep original logic but add fuzzy for edge cases
    if any(
        k in value
        for k in [
            "cannot move",
            "can't move",
            "cannot relocate",
            "can't relocate",
            "must stay",
            "stay home",
            "strict local",
        ]
    ):
        return "strict_local"
    if any(k in value for k in ["stay close", "near home", "nearby", "close to home", "local", "prefer local"]):
        return "nearby_ok"
    if any(
        k in value
        for k in [
            "anywhere",
            "all italy",
            "across italy",
            "move anywhere",
            "relocate anywhere",
            "can relocate",
            "could relocate",
            "willing to relocate",
        ]
    ):
        return "italy_ok"
    # fuzzy fallback
    if difflib.SequenceMatcher(None, value, "cannot move").ratio() > 0.8:
        return "strict_local"
    if difflib.SequenceMatcher(None, value, "willing to relocate").ratio() > 0.8:
        return "italy_ok"
    return None


# entity extraction
class EntityExtractor:
    def __init__(self):
        self.model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        # A simpler regex fallback for common patterns
        self.regex_patterns = {
            "REGION": re.compile(
                r"\b(north|northern|nord|center|central|centro|south|southern|sud|island|islands|sicily|sardinia|sicilia|sardegna|"
                r"milano|roma|napoli|torino|bologna|firenze|genova|venezia|padova|bari|cagliari|palermo|catanzaro|l'aquila|"
                r"teramo|pescara|chieti|ancona|perugia|pisa|siena|trento|trieste|udine|modena|parma|ferrara|ravenna|"
                r"rimini|forli|cesena|piacenza|cremona|mantova|bergamo|brescia|como|varese|sondrio|lecco|asti|alessandria|"
                r"cuneo|novara|verbania|biella|ivrea|domodossola|viterbo|rieti|frosinone|latina|terni|orvieto|spoleto|"
                r"civitanova|salerno|foggia|lecce|brindisi|taranto|potenza|matera|crotone|cosenza|reggio calabria|"
                r"vibo valentia|campobasso|isernia|avellino|benevento|caserta|cezze|siracusa|messina|trapani|agrigento|"
                r"ragusa|sassari|nuoro|oristano)\b",
                re.I,
            ),
            "FIELD_OF_INTEREST": re.compile(
                r"\b(computer science|informatics|ai|artificial intelligence|data science|machine learning|software engineering|"
                r"programming|cybersecurity|information technology|computer engineering|civil engineering|mechanical engineering|"
                r"electronic engineering|industrial engineering|biomedical engineering|materials engineering|telecommunications|"
                r"physics|mathematics|chemistry|biology|biotechnology|environmental science|geology|astronomy|statistics|"
                r"medicine|surgery|pharmacy|nursing|physiotherapy|dentistry|veterinary|nutrition|psychology|social work|"
                r"sociology|education|pedagogy|philosophy|history|literature|linguistics|law|economics|management|marketing|"
                r"political science|international relations|architecture|design|urban planning|arts|fine arts|music|cinema|"
                r"theatre|cultural heritage|museology|conservation|visual arts|graphic design|fashion design|"
                r"ingegneria|informatica|fisica|matematica|chimica|biologia|biotecnologia|medicina|farmacia|psicologia|"
                r"giurisprudenza|economia|architettura|lettere|filosofia|scienze della formazione|scienze politiche)\b",
                re.I,
            ),
            "ACADEMIC_BACKGROUND": re.compile(
                r"\b(high school|secondary school|scuola superiore|liceo scientifico|liceo classico|liceo linguistico|"
                r"liceo artistico|liceo musicale|istituto tecnico|istituto professionale|bachelor|bachelor's degree|"
                r"laurea triennale|master|master's degree|laurea magistrale|phd|doctorate|dottorato|"
                r"engineering degree|diploma di laurea|specializzazione|post-laurea)\b",
                re.I,
            ),
            "CAREER_GOAL": re.compile(
                r"\b(research|academic|phd|scientist|researcher|industry|business|company|corporate|"
                r"software engineer|developer|doctor|medical|physician|psychologist|therapist|teacher|professor|"
                r"school|teaching|consulting|consultant|advisor|public sector|civil service|government|administration|"
                r"politics|lawyer|solicitor|barrister|judge|engineer|architect|designer|entrepreneur|startup|"
                r"management|healthcare|hospital|ricerca|docenza|consulenza|libera professione|sanità)\b",
                re.I,
            ),
            "MOBILITY_PREFERENCE": re.compile(
                r"\b(cannot move|can't move|cannot relocate|can't relocate|must stay|stay home|strict local|"
                r"stay close|near home|nearby|close to home|local|prefer local|anywhere|all italy|across italy|"
                r"move anywhere|relocate anywhere|can relocate|could relocate|willing to relocate|"
                r"non posso spostarmi|voglio restare vicino|disponibile a trasferirmi|ovunque in italia)\b",
                re.I,
            ),
        }

    def extract(self, text: str) -> dict:
        # Primary: GLiNER with lower threshold
        entities = self.model.predict_entities(text, ENTITY_DESCRIPTIONS, threshold=0.45)
        extracted = {}
        for ent in entities:
            label = ent["label"]
            extracted.setdefault(label, []).append({"text": ent["text"], "score": float(ent["score"])})

        # Fallback: regex if GLiNER missed some labels
        for label, pattern in self.regex_patterns.items():
            if label not in extracted:
                matches = pattern.findall(text)
                if matches:
                    # use first match with a moderate score
                    extracted[label] = [{"text": matches[0], "score": 0.6}]

        return extracted


_extractor_instance = None


def get_extractor():
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = EntityExtractor()
    return _extractor_instance


entity_verification_llm = get_llm()  # stronger model

FEW_SHOT_EXAMPLES = """
Example 1:
User: "My name is Marco. I am a student from Milan, I want to study computer science."
Output: {"name": "Marco", "region": "north", "field_of_interest": "computer science", "academic_background": null, "mobility_preference": null, "career_goal": null}

Example 2:
User: "I have a bachelor in biology and want to work in research, preferably near Rome."
Output: {"name": null, "region": "center", "field_of_interest": "biology", "academic_background": "bachelor's degree", "mobility_preference": "nearby_ok", "career_goal": "research"}

Example 3:
User: "I cannot move, I want to study arts."
Output: {"name": null, "region": null, "field_of_interest": "arts", "academic_background": null, "mobility_preference": "strict_local", "career_goal": null}

Example 4:
User: "I'm finishing my master's in mechanical engineering and I'm really into renewable energy."
Output: {"name": null, "region": null, "field_of_interest": "engineering", "academic_background": "master's degree", "mobility_preference": null, "career_goal": null}

Example 5:
User: "I'm from Sicily, I can't leave the island for university."
Output: {"name": null, "region": "islands", "field_of_interest": null, "academic_background": null, "mobility_preference": "strict_local", "career_goal": null}

Example 6:
User: "My name is Giulia. I'm a student from Naples, I have a bachelor's degree in psychology, and I want to work in clinical practice. I can move anywhere in Italy."
Output: {"name": "Giulia", "region": "south", "field_of_interest": "psychology", "academic_background": "bachelor's degree", "mobility_preference": "italy_ok", "career_goal": "clinical practice"}

Example 7:
User: "I'm from Bologna and I want to study environmental science."
Output: {"name": null, "region": "north", "field_of_interest": "environmental science", "academic_background": null, "mobility_preference": null, "career_goal": null}

Example 8:
User: "I have a PhD in chemistry and I want to stay in academia."
Output: {"name": null, "region": null, "field_of_interest": null, "academic_background": "phd", "mobility_preference": null, "career_goal": "research"}
"""


def verify_and_complete_with_llm(
    text: str, gliner_entities: dict, profile_state: UserProfileState, dialog_context: str = ""
):
    prompt = f"""You are an expert slot filler for a university counselling system.
Return a JSON object with these keys: gender, region, field_of_interest, academic_background, mobility_preference, career_goal.
Use null if a slot is not mentioned. Normalise values to standard forms (e.g., 'north' for Milan).

Here are some examples:
{FEW_SHOT_EXAMPLES}

Now analyse the following:
User text:
{text}

Recent dialogue:
{dialog_context}

GLiNER hints:
{json.dumps(gliner_entities, ensure_ascii=False)}

Current profile:
{json.dumps(profile_state.profile, ensure_ascii=False)}

Output ONLY the JSON object, nothing else."""
    try:
        msg = entity_verification_llm.invoke(prompt)
        raw = msg.content if hasattr(msg, "content") else str(msg)
        parsed = parse_slot_output(raw)
        return parsed.model_dump()
    except Exception as e:
        print("LLM verification error:", e)
        return {}


PROFILE_SLOTS = (
    "gender",
    "region",
    "field_of_interest",
    "academic_background",
    "mobility_preference",
    "career_goal",
)


def best_entity_text(entity_items: Any) -> Tuple[str | None, float]:
    """
    entity_items is expected to be a list like:
      [{"text": "...", "score": 0.91}, ...]
    """
    if not entity_items:
        return None, 0.0

    if isinstance(entity_items, dict):
        text = entity_items.get("text")
        score = float(entity_items.get("score", 0.0))
        return text, score

    best = max(entity_items, key=lambda x: float(x.get("score", 0.0)))
    return best.get("text"), float(best.get("score", 0.0))


def update_profile_from_text(
    text: str, profile_state: UserProfileState, dialogue_tracker: DialogueTurnsTracker
) -> dict:
    gliner_entities = get_extractor().extract(text)
    context = dialogue_tracker.get_recent_context() if dialogue_tracker else ""

    llm_updates = verify_and_complete_with_llm(text, gliner_entities, profile_state, context)
    print("\nLLM verification result:", llm_updates)

    # Heuristic updates from GLiNER (unchanged loop, but using improved normalisers)
    heuristic_updates = {}
    for gliner_label, slot_name, normalizer in [
        # ("GENDER", "gender", normalize_gender),
        ("REGION", "region", normalize_region),
        ("FIELD_OF_INTEREST", "field_of_interest", normalize_field_of_interest),
        ("ACADEMIC_BACKGROUND", "academic_background", normalize_academic_background),
        ("CAREER_GOAL", "career_goal", normalize_career_goal),
        ("MOBILITY_PREFERENCE", "mobility_preference", normalize_mobility),
    ]:
        if gliner_label in gliner_entities:
            ent_text, score = best_entity_text(gliner_entities[gliner_label])
            val = normalizer(ent_text)
            if val is not None:
                heuristic_updates[slot_name] = (val, score, "gliner")

    print(f"Current entities heuristic : {heuristic_updates}\n\n")
    print(f"Current entities llmupdates : {llm_updates}\n\n")

    # Apply heuristic first
    for slot, (val, score, source) in heuristic_updates.items():
        profile_state.update_slot(slot, val, score, source)

    # Apply LLM results
    for slot in PROFILE_SLOTS:
        raw_val = llm_updates.get(slot)
        if raw_val is None:
            continue
        val = raw_val if not isinstance(raw_val, dict) else raw_val.get("value", raw_val)
        score = 0.65
        if isinstance(raw_val, dict) and "confidence" in raw_val:
            score = float(raw_val["confidence"])

        # Normalise again with the same functions
        normalizer = NORMALIZERS.get(slot)
        if normalizer is not None:
            val = normalizer(val)

        if val is not None:
            profile_state.update_slot(slot, val, score, "llm")

    print("\n**Entity Debug**")
    print("User Utterance:")
    print(text)
    print("\nRECOGNIZED ENTITIES (GLiNER):")
    print(json.dumps(gliner_entities, indent=2, ensure_ascii=False))
    print("\nLLM SLOT CANDIDATES:")
    print(json.dumps(llm_updates, indent=2, ensure_ascii=False))

    print("\nUPDATED PROFILE:")
    print(json.dumps(profile_state.profile, indent=2, ensure_ascii=False))
    print("\nSLOT CONFIDENCE:")
    print(json.dumps(profile_state.confidence, indent=2, ensure_ascii=False))

    return {
        "profile": profile_state.profile,
        "confidence": profile_state.confidence,
        "sources": profile_state.sources,
        "gliner_entities": gliner_entities,
        "llm_updates": llm_updates,
    }


def check_profile_completeness(profile_state: UserProfileState):
    """Check whether all required user profile slots are filled."""
    missing = []
    for slot in SLOT_ORDER:
        if REQUIRED_SLOTS[slot]["required"] and profile_state.profile[slot] is None:
            missing.append(slot)

    return {
        "complete": len(missing) == 0,
        "missing_slots": missing,
        "profile": profile_state.profile,
        "confidence": profile_state.confidence,
    }


def get_counseling_policy():
    """Return the counseling policy, slot order, and required profile fields."""
    return {
        "goal": "Collect student profile and recommend suitable university departments.",
        "required_slots": REQUIRED_SLOTS,
        "slot_order": SLOT_ORDER,
        "style": "Ask one concise question at a time.",
        "rag": "Use retrieved university information to generate grounded recommendations.",
    }


# OOD Detection
class OutOfDomainGuardOrchestrator(BaseOrchestrator):

    def __init__(self, dialogue_tracker, model=None):

        super().__init__()

        self.dialogue_tracker = dialogue_tracker

        self.model = model

    def instruct(self, dialog, utterance):

        if utterance is None:
            return None

        last_system = self.dialogue_tracker.get_last_system_turn()

        judge = LLMJudgeYesNo(
            f"""
Is the user's latest request unrelated to:

- university studies
- degree selection
- academic interests
- educational background
- university mobility
- career counseling

Consider carefully what the system previously asked:

{last_system}
""",
            reason=True,
            model=self.model,
        )

        try:
            result = judge.judge(utterance)

            if result.positive:
                return "Politely explain that you only assist " "with university and career counseling."

        except Exception as e:
            print("OOD ERROR:", e)

        return None


def infer_university_region(university_name: str, source_text: str) -> str | None:
    """
    Infer the macro-region of a university.

    Priority:
    1. Explicit REGION field in the document, if available.
    2. City-name lookup from the university name.
    3. City-name lookup from the document text.

    Returns:
        "north", "center", "south", "islands", or None.
    """

    # --------------------------------------------------------
    # 1. Look for an explicit REGION field in the source file.
    # --------------------------------------------------------
    if source_text:
        region_match = re.search(
            r"REGION:\s*(.+?)(?:\n|$)",
            source_text,
            re.IGNORECASE,
        )

        if region_match:
            explicit_region = region_match.group(1).strip().lower()

            if explicit_region in config.VALID_REGIONS:
                return explicit_region

            # Allow common variants.
            normalized = normalize_region(explicit_region)

            if normalized in config.VALID_REGIONS:
                return normalized

    # --------------------------------------------------------
    # 2. Try the university name.
    # --------------------------------------------------------
    university_lower = _safe_strip(university_name).lower()

    for city, region in CITY_TO_REGION.items():
        if city in university_lower:
            return region

    # --------------------------------------------------------
    # 3. Try the source document as a fallback.
    #
    # This is useful when the UNIVERSITY field itself does not
    # explicitly contain the city but another metadata field does.
    # --------------------------------------------------------
    if source_text:
        source_lower = source_text.lower()

        for city, region in CITY_TO_REGION.items():
            if city in source_lower:
                return region

    return None


@dataclass(slots=True)
class DocumentChunk:
    text: str
    source: str
    chunk_id: int
    university: str | None = None
    course: str | None = None
    url: str | None = None
    region: str | None = None


def extract_course_names_from_text(text: str) -> List[str]:
    """
    Extract official course/department names from a chunk of text.
    Uses patterns like:
      - "COURSE: [CODE] NAME"
      - "Bachelor's Degree in NAME"
      - "Master's Degree in NAME"
      - "LM-XX - NAME"
      - "Degree Program in NAME"
      - "CURRICULUM: NAME"
    """
    names = set()
    # Pattern for COURSE lines
    course_pattern = re.compile(r"COURSE:\s*(?:\[.*?\]\s*)?([^\n]+)", re.I)
    # Pattern for degree lines
    degree_pattern = re.compile(r"(?:Bachelor|Master)(?:'s)?\s+Degree(?:\s+Program)?\s+in\s+([^\n,.]+)", re.I)
    # Pattern for LM-XX lines
    lm_pattern = re.compile(r"LM-\d+\s*[-–]\s*([^\n]+)", re.I)
    # Pattern for "Degree Program" lines
    prog_pattern = re.compile(r"Degree Program\s+in\s+([^\n]+)", re.I)
    # Pattern for "CURRICULUM:" lines
    cur_pattern = re.compile(r"CURRICULUM:\s*([^\n]+)", re.I)

    for pattern in [course_pattern, degree_pattern, lm_pattern, prog_pattern, cur_pattern]:
        for match in pattern.findall(text):
            name = match.strip()
            if name and len(name) > 3:  # avoid noise
                # Clean up: remove leading/trailing punctuation, extra spaces
                name = re.sub(r"\s+", " ", name)
                names.add(name)
    return list(names)


def normalize_embeddings(vectors: np.ndarray):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)

    return vectors / norms


# FAISS RAG retrieval
class UniversityRAGRetriever:

    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.documents = []

        # Global FAISS index =>
        # Used when the student is allowed to study anywhere
        # in Italy or when no geographic restriction is known.
        self.index = None

        # Region-specific FAISS indices.
        #
        # IMPORTANT:
        # The integer values stored in these indices correspond to
        # positions in self.documents.
        self.region_indices = {}

        self.build_index()

    # Load documents
    def load_documents(self):

        docs = []
        for fp in glob.glob(f"{DATA_DIR}/*.txt"):
            with open(fp, "r", encoding="utf-8") as f:
                docs.append((fp, f.read()))

        return docs

    # Chunk documents
    def chunk_documents(self, docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
        """
        Sentence-aware chunking using LangChain.

        chunk_size:
            Maximum characters per chunk.
        chunk_overlap:
            Characters retained between neighboring chunks.

        Returns:
            List[DocumentChunk]
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                ": ",
                ", ",
                " ",
            ],
            keep_separator=True,
            length_function=len,
        )

        chunks = []
        for source, text in docs:
            # Parse metadata
            uni_match = re.search(r"UNIVERSITY:\s*(.*?)(?:\n|$)", text)
            course_match = re.search(r"COURSE:\s*(.*?)(?:\n|$)", text)
            url_match = re.search(r"URL:\s*(.*?)(?:\n|$)", text)
            # Infer university location independently from the student's
            # home region. This is DOCUMENT metadata used for retrieval.
            university = uni_match.group(1).strip() if uni_match else "Unknown"
            region = infer_university_region(university_name=university, source_text=text)
            course = course_match.group(1).strip() if course_match else "Unknown"
            url = url_match.group(1).strip() if url_match else "Unknown"

            split_texts = splitter.split_text(text)
            for idx, chunk_text in enumerate(split_texts):
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        source=source,
                        chunk_id=idx,
                        university=university,
                        course=course,
                        url=url,
                        region=region,
                    )
                )
        return chunks

    def build_index_signature(self) -> str:
        config = {
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "data_dir": config.DATA_DIR,
        }

        payload = json.dumps(
            config,
            sort_keys=True,
        ).encode("utf-8")

        return hashlib.sha256(payload).hexdigest()[:16]

    # Build cosine-similarity index
    def build_index(self):
        # load from disk if available, otherwise build from scratch
        cache_path = os.path.join(config.INDEX_CACHE_DIR, f"index_{self.build_index_signature()}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.documents = data["documents"]
            self.index = data["index"]
            self.region_indices = data["region_indices"]
            print("Loaded FAISS index from cache.")
            return

        raw_docs = self.load_documents()
        if not raw_docs:
            raise RuntimeError(f"No documents found in {config.DATA_DIR}")

        self.documents = self.chunk_documents(raw_docs)
        if not self.documents:
            raise RuntimeError("Document chunking produced no chunks.")

        texts = [d.text for d in self.documents if d.text.strip()]
        if not texts:
            raise RuntimeError("All chunks are empty.")

        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        if len(embeddings) == 0:
            raise RuntimeError("Embedding generation failed.")

        embeddings = normalize_embeddings(embeddings)
        dimension = embeddings.shape[1]

        # build global FAISS index.
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

        # Build REGION-SPECIFIC FAISS indices
        # This lets us perform retrieval directly inside the geographic
        # constraint instead of retrieving globally and filtering later.
        self.region_indices = {}
        for region in VALID_REGIONS:

            region_doc_indices = [idx for idx, doc in enumerate(self.documents) if doc.region == region]
            if not region_doc_indices:
                continue

            region_embeddings = embeddings[region_doc_indices]

            region_index = faiss.IndexFlatIP(dimension)
            region_index.add(region_embeddings.astype(np.float32))

            self.region_indices[region] = {
                "index": region_index,
                "document_indices": region_doc_indices,
            }

            print(f"Built FAISS region index: " f"{region} -> {len(region_doc_indices)} chunks")

        # save built index to cache
        with open(cache_path, "wb") as f:
            pickle.dump({"documents": self.documents, "index": self.index, "region_indices": self.region_indices}, f)
        print("Saved FAISS index to cache.")

    def _retrieve_by_regions(self, query_embedding, allowed_regions: set[str] | None, top_k: int):
        """
        Retrieve chunks according to the geographic policy.

        allowed_regions:
            None -> use the global index
            {"north"} -> retrieve only from North
            {"north", "center"} -> retrieve from North + Center
            all four regions -> effectively Italy-wide
        """

        # No geographic restriction
        if allowed_regions is None:
            scores, indices = self.index.search(query_embedding, top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue

                results.append((self.documents[idx], float(score)))

            return results

        # Geographic restriction
        # Retrieve from every permitted region independently.
        regional_results = []
        for region in allowed_regions:

            region_data = self.region_indices.get(region)
            if region_data is None:
                continue

            region_index = region_data["index"]
            document_indices = region_data["document_indices"]

            regional_scores, regional_indices = region_index.search(query_embedding, top_k)
            for score, local_idx in zip(regional_scores[0], regional_indices[0]):

                if local_idx < 0:
                    continue

                global_idx = document_indices[local_idx]
                regional_results.append((self.documents[global_idx], float(score)))

        # Sort all geographically-valid results by similarity.
        regional_results.sort(key=lambda x: x[1], reverse=True)

        return regional_results[:top_k]

    # Natural-language profile query
    def build_query(self, profile):

        return f"""
Student's gender: 
{profile.get('gender')}.
        
Student interested in
{profile.get('field_of_interest')}.

Student's academic background:
{profile.get('academic_background')}.

Student lives in the:
{profile.get('region')}.

Student's mobility preference:
{profile.get('mobility_preference')}.

Recommend suitable university departments and degree programs considering the aforementioned user profile.
"""

    # Retrieve + rerank
    def retrieve(
        self,
        profile: dict,
        top_k: int = config.TOP_K,
        final_k: int = config.FINAL_K,
        allowed_regions_override: set[str] | None = None,
    ):

        # profile based query for semantic retrieval
        query = self.build_query(profile)

        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = normalize_embeddings(query_embedding)

        # geographic filtering based on user's home region and mobility preference
        if allowed_regions_override is not None:
            allowed_regions = allowed_regions_override
        else:
            allowed_regions = get_allowed_study_regions(
                home_region=profile.get("region"),
                mobility_preference=profile.get("mobility_preference"),
            )

        semantic_candidates = self._retrieve_by_regions(
            query_embedding=query_embedding,
            allowed_regions=allowed_regions,
            top_k=top_k * 2,
        )
        candidates = [chunk for chunk, _ in semantic_candidates]

        field = profile.get("field_of_interest")
        if field:
            lexical_candidates = [chunk for chunk in candidates if field.casefold() in chunk.text.casefold()]

            if len(lexical_candidates) >= final_k:
                candidates = lexical_candidates

        if not candidates:
            return []

        rerank_pairs = [(query, chunk.text) for chunk in candidates]
        rerank_scores = self.reranker.predict(rerank_pairs)

        reranked = sorted(zip(candidates, rerank_scores), key=lambda item: float(item[1]), reverse=True)

        return reranked[:final_k]


_retriever_instance = None


def get_retriever():

    global _retriever_instance
    if _retriever_instance is None:

        print("Building university retriever...")
        _retriever_instance = UniversityRAGRetriever()

    return _retriever_instance


class RecommendedDepartment(BaseModel):
    name: str
    why_it_matches: str
    source_files: list[str] = []
    url: str | None = None


class Recommendations(BaseModel):
    departments: list[RecommendedDepartment]


def count_unique_course_candidates(retrieval_results: list[tuple[DocumentChunk, float]]) -> int:
    """
    Count distinct official courses/departments represented
    in the retrieved chunks.
    """

    names = set()

    for chunk, _ in retrieval_results:
        for name in extract_course_names_from_text(chunk.text):
            names.add(name.casefold().strip())

    return len(names)


def retrieve_universities(profile: dict) -> dict:
    """Return grounded recommendations after strict hallucination checks."""

    retriever = get_retriever()

    # retrieve based on user's profile
    retrieval_results = retriever.retrieve(profile)

    geography_relaxed = False

    if count_unique_course_candidates(retrieval_results) < 3:
        # if recommendations less than 3 => relax geographic constraint
        retrieval_results = retriever.retrieve(
            profile,
            allowed_regions_override=config.VALID_REGIONS,
        )

        geography_relaxed = True

    all_candidates = []
    for chunk, _ in retrieval_results:
        all_candidates.extend(extract_course_names_from_text(chunk.text))
    # Remove duplicates, preserve order
    candidate_map = {}
    for name in all_candidates:

        key = name.casefold().strip()
        if key not in candidate_map:
            candidate_map[key] = name
    unique_candidates = list(candidate_map.values())

    if not unique_candidates:
        # No candidates found – fallback to a message asking for more detail.
        return {
            "recommendations": "I couldn't find any specific courses matching your profile in the available documents. Could you refine your interests or background?",
            "retrieval_results": retrieval_results,
            "validated_departments": [],
        }

    prompt = f"""You are a university career counsellor. You must recommend courses ONLY from the provided candidate list.
    The candidate list contains the official names of courses/departments extracted from university documents.

    Student Profile:
    {json.dumps(profile, indent=2)}

    Candidate Courses (select from these only):
    {json.dumps(unique_candidates, indent=2)}

    Task:
    Select up to 3 courses from the candidate list that best match the student's profile. For each selected course, provide:
    - "name": the exact course name as it appears in the candidate list.
    - "why_it_matches": a short explanation based on the student's profile (e.g., "matches your interest in X and your background in Y").

    If none of the candidates are a good match, return an empty list.

    Output JSON with this structure:
    {{ "departments": [ {{ "name": "...", "why_it_matches": "..." }} ] }}
    """

    # ----- a dedicated LLM call (not the Agent) for structured output
    advisor_llm = get_llm(format="json")
    raw_response = advisor_llm.invoke(prompt)
    response_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

    # Parse the structured output
    try:
        recs = Recommendations.model_validate_json(response_text)
    except Exception:
        # Fallback: try to repair and parse again
        try:
            repaired = repair_json(response_text)
            recs = Recommendations.model_validate_json(repaired)
        except Exception as e:
            print("Failed to parse recommendations:", e)
            return {
                "recommendations": "I'm sorry, I couldn't generate precise recommendations right now.",
                "retrieval_results": retrieval_results,
            }

    # ----- Factual grounding verification -----
    validated_departments = []
    for dept in recs.departments:

        if dept.name not in unique_candidates:
            print(f"Rejected hallucinated department: " f"{dept.name}")
            continue

        source_files = []
        urls = []
        for chunk, _ in retrieval_results:

            if dept.name not in chunk.text:
                continue

            # Collect every supporting source.
            source_files.append(os.path.basename(chunk.source))

            # Collect every available URL.
            if chunk.url and chunk.url != "Unknown":
                urls.append(chunk.url)

        dept.source_files = sorted(set(source_files))
        # Use one canonical URL for display.
        dept.url = next(iter(dict.fromkeys(urls)), None)

        validated_departments.append(dept)

    # Build a human‑readable response from the validated list
    if not validated_departments:
        msg = (
            "I'm sorry, but I was unable to find enough reliable information "
            "in the university documents. Please try again or adjust your preferences."
        )
    else:
        if geography_relaxed:
            lines = [
                "I found fewer than three suitable programs within your "
                "preferred geographic area, so I expanded the search to "
                "other parts of Italy to provide additional options.",
                "",
                "Here are the recommendations:",
            ]
        else:
            lines = ["Here are recommended university departments based on your profile:"]
        for i, dept in enumerate(validated_departments, 1):
            lines.append(f"{i}. **{dept.name}**")
            lines.append(f"   {dept.why_it_matches}")
            if dept.url:
                lines.append(f"   More info: {dept.url}")
            lines.append("")
        msg = "\n".join(lines)

    return {
        "recommendations": msg,
        "retrieval_results": retrieval_results,
        "validated_departments": validated_departments,
    }


# slot-filling & user profile completeness orchestrator
class ProfileCollectionOrchestrator(BasePersistentOrchestrator):
    def __init__(self, dialogue_tracker: DialogueTurnsTracker):
        super().__init__()
        self.profile_state = UserProfileState()
        self.dialogue_tracker = dialogue_tracker
        self.finished = False
        self.started = False
        self._boot_message_sent = False

    def reset(self):
        super().reset()
        self.finished = False
        self.started = False
        self._boot_message_sent = False
        self.profile_state.reset()

    def get_welcome_message(self):
        if not self._boot_message_sent:
            self._boot_message_sent = True
            self.started = True
            self.profile_state.reset()
            return (
                "Hello! I help students choose university programs based on their background, interests, goals and constraints.\n\n"
                "Would you intriduce yourself by telling me your first name and what is your current educational background?"
            )
        return None

    def evaluate_profile_completeness(self):
        """evaluates profile completeness"""
        return check_profile_completeness(self.profile_state)

    def recommend(self):
        """return the recommendations based on the retrieved universities"""
        return retrieve_universities(self.profile_state.profile)

    def lowest_confidence_filled_slot(self):
        candidates = [
            slot
            for slot in SLOT_ORDER
            if self.profile_state.profile.get(slot) is not None
            and self.profile_state.confidence.get(slot, 0.0) < CONFIRMATION_THRESHOLD
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda s: self.profile_state.confidence.get(s, 0.0))

    def generate_question(self, missing_slots):
        """
        Determines which slot(s) to ask about next, based on SLOT_ORDER and
        the list of missing slots. Returns a string starting with "[REPHRASE] "
        so that the Agent will rephrase it naturally.
        """
        # --- First, handle common combinations for efficiency ---
        # If both academic_background and field_of_interest are missing,
        # ask them together (they are often mentioned together).
        if "academic_background" in missing_slots and "field_of_interest" in missing_slots:
            hint = (
                "Ask the student about their current educational background "
                "(e.g., high school, bachelor's, master's) and which study areas interest them most."
            )
            return f"[REPHRASE] {hint}"

        # If both region and mobility_preference are missing, ask them together.
        if "region" in missing_slots and "mobility_preference" in missing_slots:
            hint = (
                "Ask the student which area of Italy they live in and whether they are willing "
                "to relocate for university studies."
            )
            return f"[REPHRASE] {hint}"

        # --- Otherwise, ask the first missing slot according to SLOT_ORDER ---
        for slot in SLOT_ORDER:
            if slot in missing_slots:
                # Mapping from slot to hint content
                slot_hints = {
                    "name": ("Ask the student for their first name."),
                    "academic_background": (
                        "Ask the student about their current educational background "
                        "(e.g., high school, bachelor's, master's, PhD)."
                    ),
                    "field_of_interest": (
                        "Ask the student which subjects or study areas interest them most "
                        "(e.g., computer science, medicine, engineering, arts)."
                    ),
                    "career_goal": (
                        "Ask the student what they would like to do professionally after graduation "
                        "(e.g., research, industry, teaching, clinical practice)."
                    ),
                    "region": (
                        "Ask the student which area of Italy they currently live in "
                        "(e.g., north, center, south, islands)."
                    ),
                    "mobility_preference": (
                        "Ask the student if they are willing to relocate for university studies "
                        "(e.g., stay local, willing to move, can go anywhere)."
                    ),
                }
                hint = slot_hints[slot]
                return f"[REPHRASE] {hint}"

    def instruct(self, dialog, utterance):
        if utterance is None or utterance.strip() == "":
            if not self.started:
                self.started = True
                self.profile_state.reset()
                return (
                    "Hello! I help students choose university programs based on their background, interests, goals and constraints.\n\n"
                    "What is your current educational background and which study areas interest you most?"
                )

        if self.finished:
            return None

        if utterance:
            update_profile_from_text(
                utterance,
                profile_state=self.profile_state,
                dialogue_tracker=self.dialogue_tracker,
            )

        low_conf_slot = self.lowest_confidence_filled_slot()
        if low_conf_slot is not None and self.profile_state.profile.get(low_conf_slot) is not None:
            pretty = low_conf_slot.replace("_", " ")
            return f"Just to confirm, I understood your {pretty} as '{self.profile_state.profile[low_conf_slot]}'. Is that correct?"

        evaluation = self.evaluate_profile_completeness()
        if evaluation["complete"]:
            self.finished = True
            rag_result = self.recommend()
            return "All profile information is collected.\n\n" + str(rag_result["recommendations"])

        missing_slots = evaluation["missing_slots"]
        if missing_slots:
            return self.generate_question(missing_slots)

        return None


class SecureLengthOrchestrator(LengthOrchestrator):
    def instruct(self, dialog, utterance):
        if dialog is None:
            return None
        return super().instruct(dialog, utterance)


# counselor = counselor | ood_guard | profile_guard | length_guard | finish_reflex


# run Counselor DM Server
if __name__ == "__main__":

    configure_application()

    # print("Model:", MODEL_URI)
    # print("Starting server...")

    # Server.serve(
    #     [counselor],
    #     port=1335,
    #     host="0.0.0.0",
    #     stateless=False,
    # )

    dialogue_tracker = DialogueTurnsTracker()

    ood_guard = OutOfDomainGuardOrchestrator(dialogue_tracker)

    profile_guard = ProfileCollectionOrchestrator(dialogue_tracker)

    # length orchestrator
    length_guard = SecureLengthOrchestrator(min=5, max=20)

    # finish reflex
    finish_reflex = SimpleReflexOrchestrator(
        condition=lambda utt: any(x in utt.lower() for x in ["bye", "stop", "end", "quit"]),
        instruction=(
            "Thank you for using the University Career Counseling service. I hope that I helped you. Goodbye."
        ),
    )

    # edw ein o man - wx aman
    counselor = Agent(
        persona=Persona(
            name="University Career Counselor",
            #             role=("""You are a university career counselor. You ONLY ask about the following aspects:
            #  academic_background, field_of_interest, career_goal, region (where the student lives), and mobility_preference. You NEVER ask about grades, test scores, or any other personal details. If the student asks about something else,
            #  politely redirect them to the profile questions."""),
            #         ),
            role=(
                "You are a university career counselor. When you see a message starting with '[REPHRASE]', "
                "you must rephrase that instruction into a natural, friendly question without changing its meaning. "
                "Do not ask about anything else. If the message does not start with '[REPHRASE]', you may answer normally "
                "but stick to the topic of university and career counseling."
            ),
        ),
        name="Counselor",
        think=False,
        response_details=(
            "Be concise, professional, and ask only one question at a time based on the given slot order and the missing slots."
        ),
        tools=[
            get_counseling_policy,
            profile_guard.evaluate_profile_completeness,
            profile_guard.recommend,
        ],
    )

    # have the counselor agent pass through the OOD guard, profile guard, length guard, and finish reflex orchestrators
    # counselor_agent acts as the orchestratos of the orchsetrators
    counselor = counselor | ood_guard | profile_guard | length_guard | finish_reflex

    print("\n\nUniversity Career Counselor is running...\n")

    boot = profile_guard.get_welcome_message()
    if boot:
        dialogue_tracker.add_system_turn(boot)
        print("Counselor:", boot)

    while True:
        user_text = input("User: ")
        dialogue_tracker.add_user_turn(user_text)
        if user_text.strip().lower() in ["quit", "exit", "bye bye", "bye", "goodbye", "thank you"]:
            break
        reply = counselor(user_text)
        dialogue_tracker.add_system_turn(reply)
        print("Counselor:", reply)
