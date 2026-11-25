import difflib
from typing import Any

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


def normalize_name(name: str) -> str:
    """Normalize a name to uppercase first letter and lowercase remainder."""
    if not name:
        return name
    return name.strip().capitalize()


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


MOBILITY_CANONICAL = {"strict_local", "nearby_ok", "italy_ok"}


def normalize_mobility(value: Any) -> str | None:
    value = _safe_strip(value).lower()
    if not value:
        return None

    if value in MOBILITY_CANONICAL:
        return value

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


NORMALIZERS = {
    "region": normalize_region,
    "field_of_interest": normalize_field_of_interest,
    "academic_background": normalize_academic_background,
    "career_goal": normalize_career_goal,
    "mobility_preference": normalize_mobility,
}
