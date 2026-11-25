import re
import json
from typing import Any, Tuple
from json_repair import repair_json
from langchain_core.output_parsers import PydanticOutputParser
import unicodedata
import config
import llm
import models
import normalization
from gliner import GLiNER

slot_parser = PydanticOutputParser(pydantic_object=models.SlotExtraction)

ENTITY_DESCRIPTIONS = {
    "NAME": "Il nome proprio dell'utente, se menzionato.",
    "REGION": "Macro-regione italiana in cui vive l'utente: nord, centro, sud, isole.",
    "FIELD_OF_INTEREST": "Area di studio universitaria o interesse accademico, ad esempio informatica, psicologia, arte, scienze ambientali.",
    "CAREER_GOAL": "Obiettivi professionali dopo la laurea: ricerca, industria, pratica clinica, insegnamento, consulenza, settore pubblico.",
    "ACADEMIC_BACKGROUND": "Livello di istruzione attuale o precedente: diploma di scuola superiore, liceo scientifico, laurea triennale, laurea magistrale, laurea in ingegneria.",
    "MOBILITY_PREFERENCE": "Preferenza sulla disponibilità dell'utente a trasferirsi o meno per gli studi universitari.",
}

# ────────────────────────────────────────────────────────────────
# First-name → gender lookup (Italian + common English names)
# ────────────────────────────────────────────────────────────────
# Purpose:
#   - Infer the student's gender from their first name when the
#     student never states it explicitly.
#   - Includes Italian names AND English names that Italian
#     students commonly use when speaking English
#     (e.g. "Helen" instead of "Elena", "Elizabeth" for "Elisabetta").
#
# Ambiguity note:
#   "andrea", "simone", "nicola", "michele", "gabriele",
#   "daniele", "elia" are MALE in Italian even though some of them
#   are female in English or French. Because the system is used
#   inside Italian universities, the Italian reading wins.
# ────────────────────────────────────────────────────────────────

FEMALE_NAMES: dict[str, str] = {
    # ── Italian female names (100) ─────────────────────────────
    "maria": "Mary",
    "anna": "Ann",
    "giulia": "Julia",
    "sofia": "Sophia",
    "alessandra": "Alexandra",
    "francesca": "Frances",
    "chiara": "Claire",
    "sara": "Sarah",
    "laura": "Laura",
    "martina": "Martina",
    "elena": "Helen",
    "beatrice": "Beatrice",
    "vittoria": "Victoria",
    "camilla": "Camilla",
    "giorgia": "Georgia",
    "valentina": "Valentina",
    "federica": "Frederica",
    "silvia": "Sylvia",
    "alice": "Alice",
    "emma": "Emma",
    "aurora": "Aurora",
    "ginevra": "Guinevere",
    "bianca": "Blanche",
    "matilde": "Matilda",
    "nicoletta": "Nicolette",
    "roberta": "Roberta",
    "simona": "Simona",
    "cristina": "Christina",
    "paola": "Paula",
    "lucia": "Lucy",
    "marta": "Martha",
    "rosa": "Rose",
    "angela": "Angela",
    "isabella": "Isabella",
    "caterina": "Catherine",
    "margherita": "Margaret",
    "rebecca": "Rebecca",
    "veronica": "Veronica",
    "ilaria": "Hilary",
    "noemi": "Naomi",
    "arianna": "Ariadne",
    "carlotta": "Charlotte",
    "diana": "Diana",
    "elisa": "Elise",
    "gaia": "Gaia",
    "irene": "Irene",
    "lavinia": "Lavinia",
    "ludovica": "Ludovica",
    "michela": "Michelle",
    "miriam": "Miriam",
    "monica": "Monica",
    "nadia": "Nadia",
    "patrizia": "Patricia",
    "sabrina": "Sabrina",
    "serena": "Serena",
    "stefania": "Stephanie",
    "teresa": "Teresa",
    "vanessa": "Vanessa",
    "viola": "Viola",
    "adele": "Adele",
    "agata": "Agatha",
    "agnese": "Agnes",
    "daniela": "Danielle",
    "alessia": "Alessia",
    "angelica": "Angelica",
    "antonella": "Antonella",
    "barbara": "Barbara",
    "benedetta": "Benedicta",
    "carla": "Carla",
    "cinzia": "Cynthia",
    "clara": "Clara",
    "claudia": "Claudia",
    "concetta": "Concetta",
    "daniela": "Danielle",
    "deborah": "Deborah",
    "denise": "Denise",
    "donatella": "Donatella",
    "elettra": "Electra",
    "emilia": "Emily",
    "enrica": "Henrietta",
    "erika": "Erica",
    "fabiana": "Fabiana",
    "flavia": "Flavia",
    "flora": "Flora",
    "gabriella": "Gabrielle",
    "gemma": "Gemma",
    "gianna": "Gianna",
    "giovanna": "Joanna",
    "grazia": "Grace",
    "ida": "Ida",
    "lorena": "Lorena",
    "loredana": "Loredana",
    "luisa": "Louise",
    "maddalena": "Magdalene",
    "marina": "Marina",
    "melissa": "Melissa",
    "ornella": "Ornella",
    "raffaella": "Raphaela",
    "rita": "Rita",
    "rosanna": "Rosanna",
    "silvana": "Sylvana",
    "sonia": "Sonia",
    "susanna": "Susan",
    "tiziana": "Tiziana",
    # ── English female names commonly used in Italy (50) ───────
    "helen": "Helen",
    "elizabeth": "Elizabeth",
    "sarah": "Sarah",
    "emily": "Emily",
    "jennifer": "Jennifer",
    "jessica": "Jessica",
    "karen": "Karen",
    "nancy": "Nancy",
    "lisa": "Lisa",
    "ashley": "Ashley",
    "kimberly": "Kimberly",
    "donna": "Donna",
    "carol": "Carol",
    "amanda": "Amanda",
    "sharon": "Sharon",
    "kathleen": "Kathleen",
    "amy": "Amy",
    "shirley": "Shirley",
    "brenda": "Brenda",
    "pamela": "Pamela",
    "nicole": "Nicole",
    "samantha": "Samantha",
    "katherine": "Katherine",
    "ruth": "Ruth",
    "christine": "Christine",
    "debra": "Debra",
    "rachel": "Rachel",
    "carolyn": "Carolyn",
    "janet": "Janet",
    "virginia": "Virginia",
    "heather": "Heather",
    "diane": "Diane",
    "julie": "Julie",
    "joyce": "Joyce",
    "kelly": "Kelly",
    "joan": "Joan",
    "evelyn": "Evelyn",
    "lauren": "Lauren",
    "judith": "Judith",
    "olivia": "Olivia",
    "frances": "Frances",
    "cheryl": "Cheryl",
    "megan": "Megan",
    "jane": "Jane",
    "grace": "Grace",
    "rose": "Rose",
    "mary": "Mary",
    "sophie": "Sophie",
    "chloe": "Chloe",
    "hannah": "Hannah",
    "lucy": "Lucy",
    "beth": "Beth",
    "claire": "Claire",
    "sophia": "Sophia",
}

MALE_NAMES: dict[str, str] = {
    # ── Italian male names (100) ───────────────────────────────
    "marco": "Mark",
    "luca": "Luke",
    "alberto": "Albert",
    "matteo": "Matthew",
    "alessandro": "Alexander",
    "andrea": "Andrew",  # male in Italian, female in English
    "francesco": "Francis",
    "giuseppe": "Joseph",
    "antonio": "Anthony",
    "giovanni": "John",
    "roberto": "Robert",
    "stefano": "Stephen",
    "paolo": "Paul",
    "davide": "David",
    "lorenzo": "Lawrence",
    "gabriele": "Gabriel",  # male in Italian
    "riccardo": "Richard",
    "michele": "Michael",  # male in Italian, female in English
    "nicola": "Nicholas",  # male in Italian, female in English
    "pietro": "Peter",
    "salvatore": "Salvador",
    "vincenzo": "Vincent",
    "emanuele": "Emmanuel",
    "federico": "Frederick",
    "filippo": "Philip",
    "giacomo": "James",
    "giorgio": "George",
    "leonardo": "Leonard",
    "maurizio": "Maurice",
    "massimo": "Max",
    "alberto": "Albert",
    "aldo": "Aldo",
    "angelo": "Angelo",
    "bruno": "Bruno",
    "carlo": "Charles",
    "cesare": "Caesar",
    "christian": "Christian",
    "claudio": "Claude",
    "cosimo": "Cosmo",
    "cristian": "Christian",
    "cristiano": "Christian",
    "daniele": "Daniel",  # male in Italian
    "dario": "Darius",
    "diego": "Diego",
    "domenico": "Dominic",
    "eduardo": "Edward",
    "elia": "Elias",  # male in Italian
    "enrico": "Henry",
    "enzo": "Enzo",
    "ernesto": "Ernest",
    "ettore": "Hector",
    "fabio": "Fabio",
    "fabrizio": "Fabrizio",
    "ferdinando": "Ferdinand",
    "franco": "Frank",
    "gaetano": "Gaetano",
    "gennaro": "Januarius",
    "gerardo": "Gerard",
    "gianluca": "Gianluca",
    "gianni": "Johnny",
    "gianfranco": "Gianfranco",
    "gino": "Gino",
    "giuliano": "Julian",
    "giulio": "Julius",
    "gregorio": "Gregory",
    "guido": "Guy",
    "gustavo": "Gustav",
    "ignazio": "Ignatius",
    "ivan": "Ivan",
    "jacopo": "Jacob",
    "lamberto": "Lambert",
    "leopoldo": "Leopold",
    "luigi": "Louis",
    "manuel": "Emmanuel",
    "marcello": "Marcel",
    "mario": "Mario",
    "martino": "Martin",
    "mattia": "Matthias",
    "mauro": "Maurus",
    "mirko": "Mirko",
    "norberto": "Norbert",
    "oliviero": "Oliver",
    "omar": "Omar",
    "oscar": "Oscar",
    "osvaldo": "Oswald",
    "ottavio": "Octavius",
    "pasquale": "Pascal",
    "patrizio": "Patrick",
    "pierluigi": "Pierluigi",
    "piero": "Peter",
    "primo": "Primo",
    "raffaele": "Raphael",
    "raoul": "Raoul",
    "renato": "Renatus",
    "rocco": "Rocco",
    "romeo": "Romeo",
    "ruggero": "Roger",
    "samuele": "Samuel",
    "saverio": "Xavier",
    "sebastiano": "Sebastian",
    "tommaso": "Thomas",
    "simone": "Simon",  # male in Italian, female in French
    "michelangelo": "Michelangelo",
    "francesco": "Francis",
    "raffaello": "Raphael",
    "sandro": "Sandro",
    "sergio": "Sergio",
    "silvio": "Silvius",
    "vito": "Vito",
    "vittorio": "Victor",
    "adriano": "Adrian",
    "amedeo": "Amadeus",
    "arcangelo": "Archangel",
    "arturo": "Arthur",
    "augusto": "Augustus",
    "benedetto": "Benedict",
    "bernardo": "Bernard",
    "calogero": "Calogero",
    "carmelo": "Carmelo",
    "cipriano": "Cyprian",
    "corrado": "Conrad",
    "damiano": "Damian",
    "dante": "Dante",
    "dionigi": "Dionysius",
    "egidio": "Giles",
    "emilio": "Emil",
    "eugenio": "Eugene",
    "ezio": "Ezio",
    "fausto": "Faustus",
    "filippo": "Philip",
    "giacinto": "Hyacinth",
    "gildo": "Gildo",
    "giorgio": "George",
    "girolamo": "Jerome",
    "giuseppe": "Joseph",
    "gualtiero": "Walter",
    "lino": "Lino",
    "livio": "Livy",
    "lorenzo": "Lawrence",
    "luciano": "Lucian",
    "luigi": "Louis",
    "manlio": "Manlius",
    "marco": "Mark",
    "marcellino": "Marcelino",
    "milo": "Milo",
    "nazzareno": "Nazarene",
    "nicodemo": "Nicodemus",
    "nunzio": "Nuncio",
    "pellegrino": "Peregrine",
    "quirino": "Quirinus",
    "rodolfo": "Rudolph",
    "romano": "Roman",
    "rosario": "Rosary",
    "salvo": "Salvo",
    "tancredi": "Tancred",
    "tullio": "Tullius",
    "uberto": "Hubert",
    "valerio": "Valerius",
    "virgilio": "Virgil",
    "zeno": "Zeno",
    # ── English male names commonly used in Italy (50) ─────────
    "john": "John",
    "james": "James",
    "robert": "Robert",
    "michael": "Michael",
    "william": "William",
    "david": "David",
    "richard": "Richard",
    "joseph": "Joseph",
    "thomas": "Thomas",
    "charles": "Charles",
    "daniel": "Daniel",
    "matthew": "Matthew",
    "anthony": "Anthony",
    "mark": "Mark",
    "donald": "Donald",
    "steven": "Steven",
    "paul": "Paul",
    "andrew": "Andrew",
    "joshua": "Joshua",
    "kenneth": "Kenneth",
    "kevin": "Kevin",
    "brian": "Brian",
    "george": "George",
    "edward": "Edward",
    "ronald": "Ronald",
    "timothy": "Timothy",
    "jason": "Jason",
    "jeffrey": "Jeffrey",
    "ryan": "Ryan",
    "jacob": "Jacob",
    "gary": "Gary",
    "nicholas": "Nicholas",
    "eric": "Eric",
    "jonathan": "Jonathan",
    "stephen": "Stephen",
    "larry": "Larry",
    "justin": "Justin",
    "scott": "Scott",
    "brandon": "Brandon",
    "frank": "Frank",
    "benjamin": "Benjamin",
    "gregory": "Gregory",
    "samuel": "Samuel",
    "raymond": "Raymond",
    "patrick": "Patrick",
    "alexander": "Alexander",
    "jack": "Jack",
    "dennis": "Dennis",
    "jerry": "Jerry",
    "henry": "Henry",
    "peter": "Peter",
    "walter": "Walter",
    "adam": "Adam",
    "harold": "Harold",
    "sean": "Sean",
    "austin": "Austin",
    "carl": "Carl",
    "arthur": "Arthur",
    "lawrence": "Lawrence",
    "jesse": "Jesse",
    "dylan": "Dylan",
    "luke": "Luke",
    "mark": "Mark",
    "simon": "Simon",
    "daniel": "Daniel",
}

# Combined lookup table.
# If a name appears in both dicts (e.g. some overlap), the MALE
# entry is overwritten by FEMALE in this build — but the dicts
# above were built to be disjoint, so no conflict occurs.
NAME_TO_GENDER: dict[str, str] = {
    **{name: "Male" for name in MALE_NAMES},
    **{name: "Female" for name in FEMALE_NAMES},
}


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


def infer_gender_from_name(name: Any) -> str | None:
    """
    Return "Male", "Female", or None for an Italian or English first name.
    Normalisation applied:
      - lowercase
      - strip diacritics (á → a, í → i, ...)
      - strip surrounding punctuation
      - handle multi-token names ("Maria Grazia", "Anna-Maria")
    Lookup order:
      1. Full lowercase name (exact match).
      2. First token of a whitespace-separated name.
      3. Any token of a hyphenated name.
    """
    if not name:
        return None

    text = str(name).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.strip(".,;:!?'\"()[]")

    if not text:
        return None

    if text in NAME_TO_GENDER:
        return NAME_TO_GENDER[text]

    for token in re.split(r"[\s\-]+", text):
        token = token.strip(".,;:!?'\"()[]")
        if token in NAME_TO_GENDER:
            return NAME_TO_GENDER[token]

    return None


def complete_gender_slot(profile_state: models.UserProfileState) -> bool:
    """
    Fill the `gender` slot from the collected first name.
    Called as the final step of profile collection, right before
    the RAG retrieval stage. The function is a no-op if:
      - the gender slot is already filled (respect the user's
        explicit statement), OR
      - the name slot is empty, OR
      - the name cannot be resolved against the mapping.
    Returns True when the gender slot was updated.
    """
    if profile_state.profile.get("gender") is not None:
        return False

    name = profile_state.profile.get("name")
    if not name:
        return False

    inferred = infer_gender_from_name(name)
    if inferred is None:
        return False

    profile_state.update_slot(slot="gender", value=inferred, confidence=0.85, source="name_inference")

    print(f"[gender] Inferred '{inferred}' from name " f"'{name}' (source=name_inference, confidence=0.85)")
    return True


class EntityExtractor:
    def __init__(self):
        self._model = None
        # self.model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(config.DEVICE)

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

    @property
    def model(self):
        if self._model is None:
            print("[extraction] Loading GLiNER model (this happens once)...")
            self._model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(config.DEVICE)
            self._model.eval()
            print("[extraction] GLiNER ready.")

        return self._model

    def extract(self, text: str) -> dict:
        # Primary: GLiNER with lower threshold
        entities = self.model.predict_entities(text, ENTITY_DESCRIPTIONS, threshold=0.45)
        extracted = {}
        for ent in entities:
            label = ent["label"]
            extracted.setdefault(label, []).append(
                {"text": ent["text"], "score": float(ent["score"]), "source": "gliner"}
            )

        # Fallback: regex if GLiNER missed some labels
        for label, pattern in self.regex_patterns.items():
            if label not in extracted:
                matches = pattern.findall(text)
                if matches:
                    # use first match with a moderate score
                    extracted[label] = [{"text": matches[0], "score": 0.0, "source": "regex"}]

        return extracted


_extractor_instance = None


def get_extractor():
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = EntityExtractor()
    return _extractor_instance


entity_verification_llm = llm.get_llm()  # stronger model


def parse_slot_output(llm_output: str) -> models.SlotExtraction:

    try:
        return slot_parser.parse(llm_output)

    except Exception:
        try:
            repaired = repair_json(llm_output)
            return models.SlotExtraction.model_validate_json(repaired)

        except Exception:
            return models.SlotExtraction()


FEW_SHOT_EXAMPLES = """
Esempio 1:
Utente: "Mi chiamo Marco. Sono uno studente di Milano, voglio studiare informatica."
Output: {"name": "Marco", "region": "north", "field_of_interest": "computer science", "academic_background": null, "mobility_preference": null, "career_goal": null}

Esempio 2:
Utente: "Ho una laurea triennale in biologia e voglio lavorare nella ricerca, preferibilmente vicino a Roma."
Output: {"name": null, "region": "center", "field_of_interest": "biology", "academic_background": "bachelor's degree", "mobility_preference": "nearby_ok", "career_goal": "research"}

Esempio 3:
Utente: "Non posso trasferirmi, voglio studiare arte."
Output: {"name": null, "region": null, "field_of_interest": "arts", "academic_background": null, "mobility_preference": "strict_local", "career_goal": null}

Esempio 4:
Utente: "Sto finendo la magistrale in ingegneria meccanica e mi interessa molto l'energia rinnovabile."
Output: {"name": null, "region": null, "field_of_interest": "engineering", "academic_background": "master's degree", "mobility_preference": null, "career_goal": null}

Esempio 5:
Utente: "Sono siciliano, non posso lasciare l'isola per l'università."
Output: {"name": null, "region": "islands", "field_of_interest": null, "academic_background": null, "mobility_preference": "strict_local", "career_goal": null}

Esempio 6:
Utente: "Mi chiamo Giulia. Sono una studentessa di Napoli, ho una laurea triennale in psicologia e voglio lavorare nella pratica clinica. Posso trasferirmi in tutta Italia."
Output: {"name": "Giulia", "region": "south", "field_of_interest": "psychology", "academic_background": "bachelor's degree", "mobility_preference": "italy_ok", "career_goal": "clinical practice"}

Esempio 7:
Utente: "Sono di Bologna e voglio studiare scienze ambientali."
Output: {"name": null, "region": "north", "field_of_interest": "environmental science", "academic_background": null, "mobility_preference": null, "career_goal": null}

Esempio 8:
Utente: "Ho un dottorato in chimica e voglio restare in ambito accademico."
Output: {"name": null, "region": null, "field_of_interest": null, "academic_background": "phd", "mobility_preference": null, "career_goal": "research"}
"""


def verify_and_complete_with_llm(
    text: str, gliner_entities: dict, profile_state: models.UserProfileState, dialog_context: str = ""
):
    prompt = f"""Rispondi sempre e solo in italiano, indipendentemente dalla lingua del testo seguente.

Sei un esperto di estrazione di slot informativi per un sistema di orientamento universitario.
Restituisci un oggetto JSON con queste chiavi (le chiavi devono restare in inglese, ma i valori testuali devono essere in inglese canonico quando richiesto):
gender, region, field_of_interest, academic_background, mobility_preference, career_goal.
Usa null se uno slot non è menzionato.

Normalizza i valori ai seguenti formati canonici:
- region: "north", "center", "south", "islands"
- mobility_preference: "strict_local", "nearby_ok", "italy_ok"
- academic_background: "high school", "scientific high school", "classical high school",
  "technical institute", "professional institute", "bachelor's degree", "master's degree",
  "engineering degree", "phd"
- field_of_interest: usa la forma inglese canonica (es. "computer science", "engineering",
  "psychology", "arts", "economics", "law", "medicine")
- career_goal: usa la forma inglese canonica (es. "research", "industry", "clinical practice",
  "education", "consulting", "public sector")

Ecco alcuni esempi:
{FEW_SHOT_EXAMPLES}

Analizza ora il seguente testo:
Testo utente:
{text}

Dialogo recente:
{dialog_context}

Suggerimenti da GLiNER:
{json.dumps(gliner_entities, ensure_ascii=False)}

Profilo attuale:
{json.dumps(profile_state.profile, ensure_ascii=False)}

Restituisci SOLO l'oggetto JSON, senza testo aggiuntivo e senza markdown."""
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
    "name",
    "region",
    "field_of_interest",
    "academic_background",
    "mobility_preference",
    "career_goal",
)


def update_profile_from_text(
    text: str, profile_state: models.UserProfileState, dialogue_tracker: models.DialogueTurnsTracker
) -> dict:
    gliner_entities = get_extractor().extract(text)
    context = dialogue_tracker.get_recent_context() if dialogue_tracker else ""

    llm_updates = verify_and_complete_with_llm(text, gliner_entities, profile_state, context)
    print("\nLLM verification result:", llm_updates)

    # Heuristic updates from GLiNER (unchanged loop, but using improved normalisers)
    heuristic_updates = {}
    for gliner_label, slot_name, normalizer in [
        # ("GENDER", "gender", normalization.normalize_gender),
        ("NAME", "name", normalization.normalize_name),
        ("REGION", "region", normalization.normalize_region),
        ("FIELD_OF_INTEREST", "field_of_interest", normalization.normalize_field_of_interest),
        ("ACADEMIC_BACKGROUND", "academic_background", normalization.normalize_academic_background),
        ("CAREER_GOAL", "career_goal", normalization.normalize_career_goal),
        ("MOBILITY_PREFERENCE", "mobility_preference", normalization.normalize_mobility),
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
        normalizer = normalization.NORMALIZERS.get(slot)
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


# ------------------- RIASEC part -----------------------------------
_LIKERT_FAST_MAP = {
    # Italian
    "fortemente in disaccordo": -2,
    "in disaccordo": -1,
    "neutro": 0,
    "neutrale": 0,
    "neutra": 0,
    "d'accordo": 1,
    "daccordo": 1,
    "fortemente d'accordo": 2,
    "fortementedaccordo": 2,
    # English fallbacks
    "strongly dislike": -2,
    "dislike": -1,
    "neutral": 0,
    "mixed": 0,
    "like": 1,
    "strongly like": 2,
    # Numeric strings
    "1": -2,
    "2": -1,
    "3": 0,
    "4": 1,
    "5": 2,
}


def _normalize_likert_text(text: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace."""
    t = text.strip().lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s']", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _coerce_valence(val):
    """Coerce whatever the LLM returned into int in [-2, 2] or None."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val if -2 <= val <= 2 else None
    if isinstance(val, float):
        if val.is_integer() and -2 <= val <= 2:
            return int(val)
        return None
    if isinstance(val, str):
        s = val.strip()
        if s.lower() in ("", "null", "none", "n/a", "na"):
            return None
        try:
            return _coerce_valence(int(s))
        except ValueError:
            pass
        try:
            return _coerce_valence(float(s))
        except ValueError:
            pass
        return _LIKERT_FAST_MAP.get(_normalize_likert_text(s))
    return None


# RIASEC assessment: questions, weights, scoring, LLM interpretation
RIASEC_QUESTIONS = [
    "Quanto ti piacciono le attività pratiche e manuali, come costruire, riparare o lavorare con utensili e oggetti concreti?",
    "Quanto ti piace capire come funzionano le cose, i sistemi o i fenomeni?",
    "Quanto ti piace ricercare informazioni, analizzare problemi e ragionare in modo logico?",
    "Quanto ti piace esprimere idee o emozioni attraverso forme creative come la scrittura, l'arte, la musica o il design?",
    "Quanto ti piace sperimentare e creare qualcosa di nuovo partendo da idee aperte?",
    "Quanto ti piace lavorare a stretto contatto con le persone in un ruolo di aiuto, supporto o ascolto?",
    "Quanto ti piace proporre idee, prendere l'iniziativa e organizzare progetti o attività?",
    "Quanto ti piace mantenere le attività ordinate, pianificate e sotto controllo?",
]

RIASEC_LABELS = {
    "R": "Realistico",
    "I": "Investigativo",
    "A": "Artistico",
    "S": "Sociale",
    "E": "Intraprendente",
    "C": "Convenzionale",
}

LIKERT_OPTIONS_TEXT = "Fortemente in disaccordo, In disaccordo, Neutro, D'accordo, Fortemente d'accordo"

RIASEC_WEIGHTS = [
    {"R": 1.0},
    {"I": 0.75, "R": 0.25},
    {"I": 1.0},
    {"A": 1.0},
    {"A": 0.45, "I": 0.35, "E": 0.20},
    {"S": 1.0},
    {"E": 0.75, "C": 0.25},
    {"C": 1.0},
]

RIASEC_DIMENSIONS = ["R", "I", "A", "S", "E", "C"]


def compute_riasec_scores(valences: list) -> dict:
    """
    score[c] = round( Σ_i valence[i]·weight[i][c] / (2·Σ_i weight[i][c]), 3 )

    Items whose valence is None (unanswered / uninterpretable) are excluded
    from both numerator and denominator, per the A4 decision.
    """
    scores: dict = {}
    for dim in RIASEC_DIMENSIONS:
        numerator = 0.0
        weight_sum = 0.0

        for i, weights in enumerate(RIASEC_WEIGHTS):
            if valences[i] is None:
                continue
            w = weights.get(dim, 0.0)
            if w == 0.0:
                continue
            numerator += valences[i] * w
            weight_sum += w

        scores[dim] = 0.0 if weight_sum == 0.0 else round(numerator / (2.0 * weight_sum), 3)

    return scores


def _total_dimension_weight(dim: str) -> float:
    return sum(w.get(dim, 0.0) for w in RIASEC_WEIGHTS)


def compute_riasec_profile(scores: dict) -> tuple[str, str, float]:
    """
    Returns (top3_code, description, confidence).

    Tie-breaking (B2 = option c): prefer the dimension with the *higher*
    total weight sum (the more reliable dimension), then fall back to the
    canonical RIASEC order.

    confidence = clamp(third_score - fourth_score, 0.0, 1.0)
    """
    order = {d: i for i, d in enumerate(RIASEC_DIMENSIONS)}

    ranked = sorted(RIASEC_DIMENSIONS, key=lambda d: (-scores[d], -_total_dimension_weight(d), order[d]))

    top3 = ranked[:3]
    profile_code = "".join(top3)
    labels = [RIASEC_LABELS[d] for d in top3]

    if len(labels) == 3:
        description = f"L'utente sembra essere {labels[0]}, {labels[1]} e {labels[2]}."
    elif len(labels) == 2:
        description = f"L'utente sembra essere {labels[0]} e {labels[1]}."
    elif len(labels) == 1:
        description = f"L'utente sembra essere {labels[0]}."
    else:
        description = "Dati RIASEC insufficienti."

    separation = 0.0
    if len(ranked) >= 4:
        separation = scores[ranked[2]] - scores[ranked[3]]
    confidence = max(0.0, min(1.0, separation))

    return profile_code, description, confidence


_RIASEC_LLM = None


def _get_riasec_llm():
    global _RIASEC_LLM
    if _RIASEC_LLM is None:
        _RIASEC_LLM = llm.get_llm(format="json")
    return _RIASEC_LLM


def interpret_riasec_answer(question_idx: int, answer_text: str):
    """
    Map a free-text or Likert answer to a valence in {-2,-1,0,1,2}, or None.
    """
    if not answer_text or not answer_text.strip():
        return None

    # ── Fast path: deterministic Likert / numeric parsing ────────
    normalized = _normalize_likert_text(answer_text)
    for prefix in ("sono ", "mi sento ", "direi ", "credo ", "penso ", "direi che "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break

    if normalized in _LIKERT_FAST_MAP:
        return _LIKERT_FAST_MAP[normalized]

    # ── Slow path: LLM for genuinely free-text answers ───────────
    question = RIASEC_QUESTIONS[question_idx]

    prompt = f"""Rispondi sempre e solo in italiano, indipendentemente dalla lingua del testo seguente.

Stai valutando la risposta di uno studente a una domanda RIASEC sugli interessi.

Domanda posta:
"{question}"

Risposta dello studente:
"{answer_text}"

Interpreta la risposta come un punteggio di interesse sulla seguente scala:
  -2 = Forte disinteresse
  -1 = Disinteresse
   0 = Neutro o misto
   1 = Interesse
   2 = Forte interesse

Se lo studente ha risposto con un numero da 1 a 5, interpretalo così:
  1 = Forte disinteresse, 2 = Disinteresse, 3 = Neutro, 4 = Interesse, 5 = Forte interesse.

Se la risposta è vuota, fuori tema, pone una domanda di chiarimento o non può essere interpretata come livello di interesse, restituisci null per valence.

Restituisci SOLO JSON valido esattamente in questo formato (il valore di "valence" DEVE essere un intero JSON, non una stringa né un numero decimale):
{{"valence": <intero in [-2,-1,0,1,2] oppure null>}}
"""

    try:
        response = _get_riasec_llm().invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = json.loads(repair_json(raw))

        val = parsed.get("valence") if isinstance(parsed, dict) else None
        result = _coerce_valence(val)

        if result is None:
            print(f"[riasec] Q{question_idx + 1}: could not coerce valence from {val!r} (raw: {raw!r})")

        return result

    except Exception as exc:
        print(f"[riasec] Q{question_idx + 1}: interpretation failed: {exc}")
        return None
