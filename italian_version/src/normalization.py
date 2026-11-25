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
    # High School
    "scientific high school": "scientific high school",
    "liceo scientifico": "scientific high school",
    "high-school": "high school",
    "high school": "high school",
    "scuola superiore": "high school",
    "superiori": "high school",
    "diploma": "high school",
    "maturità": "high school",
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
    "triennale": "bachelor's degree",
    "laurea": "bachelor's degree",
    "master": "master's degree",
    "MSc": "master's degree",
    "laurea magistrale": "master's degree",
    "magistrale": "master's degree",
    "specialistica": "master's degree",
    "phd": "phd",
    "dottorato": "phd",
    "doctorate": "phd",
    "dottorato di ricerca": "phd",
    "engineering degree": "engineering degree",
    "laurea in ingegneria": "engineering degree",
}

FIELD_HINTS = {
    # Computer Science & Technology
    "computer science": "computer science",
    "informatica": "computer science",
    "scienze informatiche": "computer science",
    "cs": "computer science",
    "informatics": "computer science",
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "intelligenza artificiale": "artificial intelligence",
    "data science": "data science",
    "scienza dei dati": "data science",
    "machine learning": "data science",
    "apprendimento automatico": "data science",
    "software engineering": "software engineering",
    "ingegneria del software": "software engineering",
    "programming": "computer science",
    "programmazione": "computer science",
    "cybersecurity": "computer science",
    "sicurezza informatica": "computer science",
    "information technology": "computer science",
    "tecnologie informatiche": "computer science",
    "computer engineering": "computer engineering",
    "ingegneria informatica": "computer engineering",
    "civil engineering": "engineering",
    "ingegneria civile": "engineering",
    "mechanical engineering": "engineering",
    "ingegneria meccanica": "engineering",
    "electronic engineering": "engineering",
    "ingegneria elettronica": "engineering",
    "industrial engineering": "engineering",
    "ingegneria industriale": "engineering",
    "biomedical engineering": "engineering",
    "ingegneria biomedica": "engineering",
    "materials engineering": "engineering",
    "ingegneria dei materiali": "engineering",
    "telecommunications": "engineering",
    "telecomunicazioni": "engineering",
    "ingegneria": "engineering",
    # Science
    "physics": "physics",
    "fisica": "physics",
    "mathematics": "mathematics",
    "matematica": "mathematics",
    "chemistry": "chemistry",
    "chimica": "chemistry",
    "biology": "biology",
    "biologia": "biology",
    "biotechnology": "biotechnology",
    "biotecnologia": "biotechnology",
    "biotecnologie": "biotechnology",
    "environmental science": "environmental science",
    "scienze ambientali": "environmental science",
    "geology": "geology",
    "geologia": "geology",
    "astronomy": "physics",
    "astronomia": "physics",
    "statistics": "mathematics",
    "statistica": "mathematics",
    # Medicine & Health
    "medicine": "medicine",
    "medicina": "medicine",
    "surgery": "medicine",
    "chirurgia": "medicine",
    "pharmacy": "pharmacy",
    "farmacia": "pharmacy",
    "nursing": "nursing",
    "infermieristica": "nursing",
    "physiotherapy": "physiotherapy",
    "fisioterapia": "physiotherapy",
    "dentistry": "dentistry",
    "odontoiatria": "dentistry",
    "veterinary": "veterinary",
    "veterinaria": "veterinary",
    "nutrition": "nutrition",
    "nutrizione": "nutrition",
    "psicologia": "psychology",
    "psychology": "psychology",
    "terapia occupazionale": "occupational therapy",
    "occupational therapy": "occupational therapy",
    "tecniche di radiologia": "radiology technology",
    "radiologia": "radiology technology",
    "scienze infermieristiche": "nursing",
    # Humanities & Social Sciences
    "social work": "social work",
    "servizio sociale": "social work",
    "sociology": "sociology",
    "sociologia": "sociology",
    "education": "education",
    "scienze dell'educazione": "education",
    "pedagogy": "education",
    "pedagogia": "education",
    "philosophy": "philosophy",
    "filosofia": "philosophy",
    "history": "history",
    "storia": "history",
    "literature": "literature",
    "letteratura": "literature",
    "lettere": "literature",
    "linguistics": "linguistics",
    "linguistica": "linguistics",
    "law": "law",
    "giurisprudenza": "law",
    "diritto": "law",
    "economics": "economics",
    "economia": "economics",
    "management": "management",
    "economia aziendale": "management",
    "economia e commercio": "economics",
    "marketing": "management",
    "political science": "political science",
    "scienze politiche": "political science",
    "international relations": "political science",
    "relazioni internazionali": "political science",
    "architecture": "architecture",
    "architettura": "architecture",
    "design": "design",
    "urban planning": "architecture",
    "pianificazione urbanistica": "architecture",
    "beni culturali": "cultural heritage",
    "cultural heritage": "cultural heritage",
    # Arts
    "arts": "arts",
    "arte": "arts",
    "fine arts": "arts",
    "belle arti": "arts",
    "music": "arts",
    "musica": "arts",
    "cinema": "arts",
    "theatre": "arts",
    "teatro": "arts",
    "museology": "arts",
    "museologia": "arts",
    "conservation": "arts",
    "conservazione": "arts",
    "visual arts": "arts",
    "arti visive": "arts",
    "graphic design": "design",
    "fashion design": "design",
    "design della moda": "design",
    "archeologia": "archaeology",
    "archaeology": "archaeology",
}

CAREER_HINTS = {
    "research": "research",
    "ricerca": "research",
    "accademico": "research",
    "academic": "research",
    "phd": "research",
    "scientist": "research",
    "scienziato": "research",
    "researcher": "research",
    "ricercatore": "research",
    "industry": "industry",
    "industria": "industry",
    "business": "industry",
    "azienda": "industry",
    "company": "industry",
    "corporate": "industry",
    "software engineer": "software engineering",
    "sviluppatore": "software engineering",
    "developer": "software engineering",
    "doctor": "clinical practice",
    "medico": "clinical practice",
    "medical": "clinical practice",
    "physician": "clinical practice",
    "clinica": "clinical practice",
    "pratica clinica": "clinical practice",
    "clinical practice": "clinical practice",
    "psicologo": "psychology practice",
    "psychologist": "psychology practice",
    "therapist": "psychology practice",
    "terapista": "psychology practice",
    "teacher": "education",
    "insegnante": "education",
    "professor": "education",
    "professore": "education",
    "docente": "education",
    "school": "education",
    "scuola": "education",
    "teaching": "education",
    "insegnamento": "education",
    "docenza": "education",
    "consulting": "consulting",
    "consulenza": "consulting",
    "consultant": "consulting",
    "consulente": "consulting",
    "advisor": "consulting",
    "public sector": "public sector",
    "settore pubblico": "public sector",
    "civil service": "public sector",
    "pubblica amministrazione": "public sector",
    "government": "public sector",
    "governo": "public sector",
    "administration": "public sector",
    "amministrazione": "public sector",
    "politics": "public sector",
    "politica": "public sector",
    "lawyer": "law",
    "avvocato": "law",
    "solicitor": "law",
    "barrister": "law",
    "judge": "law",
    "giudice": "law",
    "engineer": "industry",
    "ingegnere": "industry",
    "architect": "industry",
    "architetto": "industry",
    "designer": "industry",
    "entrepreneur": "industry",
    "imprenditore": "industry",
    "startup": "industry",
    "management": "industry",
    "dirigenza": "industry",
    "healthcare": "clinical practice",
    "sanità": "clinical practice",
    "hospital": "clinical practice",
    "ospedale": "clinical practice",
    "libera professione": "industry",
}


def _safe_strip(value: Any) -> str:
    """Safely strip whitespace from a string, returning an empty string for None."""
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
    """Normalize a region value to one of the canonical regions: north, center, south, islands."""
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


# For academic_background, field_of_interest, career_goal, mobility, replace the simple loop with _fuzzy_match using the existing hints:
def normalize_academic_background(value: Any) -> str | None:
    """Normalize academic background to canonical values using fuzzy matching."""
    value = _safe_strip(value).lower()
    if not value:
        return None
    return _fuzzy_match(value, BACKGROUND_HINTS, cutoff=0.85) or value  # fallback to raw


def normalize_field_of_interest(value: Any) -> str | None:
    """Normalize field of interest to canonical values using fuzzy matching."""
    value = _safe_strip(value).lower()
    if not value:
        return None
    return _fuzzy_match(value, FIELD_HINTS, cutoff=0.85) or value


def normalize_career_goal(value: Any) -> str | None:
    """Normalize career goal to canonical values using fuzzy matching."""
    value = _safe_strip(value).lower()
    if not value:
        return None
    return _fuzzy_match(value, CAREER_HINTS, cutoff=0.85) or value


MOBILITY_CANONICAL = {"strict_local", "nearby_ok", "italy_ok"}


def normalize_mobility(value: Any) -> str | None:
    """Normalize mobility preference to canonical values with fuzzy matching."""
    value = _safe_strip(value).lower()
    if not value:
        return None
    if value in MOBILITY_CANONICAL:
        return value

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
            "non posso spostarmi",
            "non posso trasferirmi",
            "non posso muovermi",
            "devo restare",
            "devo rimanere",
            "restare a casa",
            "solo nella mia regione",
            "solo in zona",
            "non voglio muovermi",
            "vincolo locale",
        ]
    ):
        return "strict_local"

    if any(
        k in value
        for k in [
            "stay close",
            "near home",
            "nearby",
            "close to home",
            "local",
            "prefer local",
            "vicino a casa",
            "zona vicina",
            "preferisco restare vicino",
            "mezza italia",
            "nord e centro",
            "regioni vicine",
            "vicino",
        ]
    ):
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
            "ovunque",
            "tutta italia",
            "tutto il paese",
            "posso trasferirmi",
            "posso muovermi",
            "disponibile a trasferirmi",
            "qualsiasi regione",
            "non ho problemi a spostarmi",
        ]
    ):
        return "italy_ok"

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
