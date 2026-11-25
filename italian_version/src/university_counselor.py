import os
import traceback
import json
import re

os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/app/.cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/app/.cache/huggingface/transformers")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/app/.cache/huggingface/sentence_transformers")
# the following are enbaled after the models have been downloaded
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

import rag.recommendations as recommendations
import extraction
import llm
import config
import models


def configure_application():
    sdialog.config.llm(f"ollama:{config.MODEL_URI}", base_url=config.OLLAMA_BASE_URL)
    sdialog.config.cache(True)

    os.makedirs(config.INDEX_CACHE_DIR, exist_ok=True)


# schema
REQUIRED_SLOTS = {
    "gender": {
        "description": "Il genere dello studente, solo se dichiarato esplicitamente.",
        "required": False,
    },
    "name": {
        "description": "Il nome proprio dello studente, solo se dichiarato esplicitamente.",
        "required": True,
    },
    "region": {
        "description": (
            "La macro-regione italiana di residenza dello studente. "
            "Deduci la macro-regione da una città, provincia o area menzionata esplicitamente. "
            "Valori canonici: north, center, south, islands."
        ),
        "required": True,
    },
    "field_of_interest": {
        "description": "La materia accademica o l'area di studio universitario che lo studente vuole studiare.",
        "required": True,
    },
    "academic_background": {
        "description": (
            "Il livello di istruzione attuale o completato dello studente, "
            "come diploma di scuola superiore, laurea triennale, laurea magistrale, "
            "laurea in ingegneria o dottorato."
        ),
        "required": True,
    },
    "mobility_preference": {
        "description": (
            "La disponibilità dello studente a trasferirsi per l'università. " "Usa strict_local, nearby_ok o italy_ok."
        ),
        "required": True,
    },
    "career_goal": {
        "description": "L'obiettivo professionale o accademico dopo la laurea.",
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


slot_parser = PydanticOutputParser(pydantic_object=models.SlotExtraction)
# format_instructions = slot_parser.get_format_instructions()


CONFIRMATION_THRESHOLD = 0.6


def check_profile_completeness(profile_state: models.UserProfileState):
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
    """Restituisce la policy di orientamento, l'ordine degli slot e i campi obbligatori del profilo dello studente."""
    return {
        "goal": "Raccogliere il profilo dello studente e consigliare dipartimenti universitari adatti.",
        "required_slots": REQUIRED_SLOTS,
        "slot_order": SLOT_ORDER,
        "style": "Poni una sola domanda concisa alla volta, sempre in italiano.",
        "rag": "Usa le informazioni universitarie recuperate per generare raccomandazioni fondate.",
    }


#####################################################################


class OutOfDomainGuardOrchestrator(BaseOrchestrator):

    def __init__(self, dialogue_tracker, model=None):
        super().__init__()

        self.dialogue_tracker = dialogue_tracker
        self.model = model

        self.judgeyesno = LLMJudgeYesNo(
            """
Rispondi sempre e solo in italiano, indipendentemente dalla lingua del testo seguente.

La richiesta più recente dell'utente è estranea ai seguenti ambiti?
- studi universitari
- scelta del corso di laurea
- interessi accademici
- background educativo
- mobilità universitaria
- orientamento professionale
- regione di residenza in Italia

Considera con attenzione cosa il sistema ha chiesto in precedenza:
{{last_system}}

Ecco l'ultima frase dell'utente:
{{utterance}}
""",
            reason=True,
            model=model,
        )

    def instruct(self, dialog, utterance):

        if utterance is None:
            return None

        last_system = self.dialogue_tracker.get_last_system_turn() or ""

        # judge_text = f"Last system message:\n{last_system}\n\n" f"User's latest utterance:\n{utterance}"
        try:
            result = self.judgeyesno.judge(dialog, last_system=last_system, utterance=utterance)
            # print("OOD result: ", result)
            if result.positive:
                return "Spiega con gentilezza che puoi assistere solo con orientamento universitario e professionale."
        except Exception as e:
            print("OOD ERROR:", repr(e))
            print("OOD ERROR TYPE:", type(e))
            traceback.print_exc()

        return None


####################################################################
# slot-filling & user profile completeness orchestrator
class ProfileCollectionOrchestrator(BasePersistentOrchestrator):
    def __init__(self, dialogue_tracker: models.DialogueTurnsTracker):
        super().__init__()

        self.profile_state = models.UserProfileState()
        self.dialogue_tracker = dialogue_tracker

        self.finished = False
        self.post_recommendation = False
        self.started = False
        self._boot_message_sent = False

        # Explicit state for the recommendation phase.
        self.profile_complete = False
        self.recommendation_ready = False
        self.recommendation_text = None

        # Used by the Agent postprocessor to force the grounded
        # recommendation text to be returned verbatim.
        self.force_recommendation_response = False

        # RIASEC phase state (runs after the user profile state is complete).
        self.riasec_completed = False
        self.current_riasec_idx = 0  # index of the question just asked [0,7] --> 8 = done
        self.riasec_goodbye_pending = False

        # Store the full RAG result so follow-up questions can be answered
        # with the same evidence that produced the recommendations.
        self.rag_result = None

        # Follow-up Q&A state (post-recommendation).
        self.followup_response_text = None
        self.force_followup_response = False

        # Force the exact RIASEC question text through postprocess.
        self.riasec_response_text = None
        self.force_riasec_response = False

    def reset(self):
        super().reset()

        self.riasec_completed = False
        self.current_riasec_idx = 0
        self.riasec_goodbye_pending = False

        self.rag_result = None
        self.followup_response_text = None
        self.force_followup_response = False

        self.finished = False
        self.post_recommendation = False
        self.started = False
        self._boot_message_sent = False

        self.profile_complete = False
        self.recommendation_ready = False
        self.recommendation_text = None
        self.force_recommendation_response = False

        # Force the exact RIASEC question text through postprocess.
        self.riasec_response_text = None
        self.force_riasec_response = False

        self.profile_state.reset()

    def get_welcome_message(self):
        if not self._boot_message_sent:
            self._boot_message_sent = True
            self.started = True
            self.profile_state.reset()

            return (
                "Ciao! Aiuto gli studenti a scegliere il percorso universitario "
                "in base al loro background, ai loro interessi, obiettivi e vincoli.\n\n"
                "Ti va di presentarti dicendomi il tuo nome e qual è il tuo "
                "attuale percorso di studi?"
            )

        return None

    def evaluate_profile_completeness(self):
        return check_profile_completeness(self.profile_state)

    def recommend(self):
        return recommendations.retrieve_universities(self.profile_state.profile, riasec=self.profile_state.riasec)

    def lowest_confidence_filled_slot(self):
        candidates = [
            slot
            for slot in SLOT_ORDER
            if self.profile_state.profile.get(slot) is not None
            and self.profile_state.confidence.get(slot, 0.0) < CONFIRMATION_THRESHOLD
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda s: self.profile_state.confidence.get(s, 0.0),
        )

    def generate_question(self, missing_slots):
        if "academic_background" in missing_slots and "field_of_interest" in missing_slots:
            hint = (
                "Chiedi allo studente il suo percorso di studi attuale "
                "(es. liceo, laurea triennale, laurea magistrale) e quali aree "
                "di studio lo interessano di più."
            )
            return f"[REPHRASE] {hint}"

        if "region" in missing_slots and "mobility_preference" in missing_slots:
            hint = (
                "Chiedi allo studente in quale area d'Italia vive e se è disposto "
                "a trasferirsi per gli studi universitari."
            )
            return f"[REPHRASE] {hint}"

        for slot in SLOT_ORDER:
            if slot in missing_slots:
                slot_hints = {
                    "name": "Chiedi allo studente il suo nome.",
                    "academic_background": (
                        "Chiedi allo studente il suo percorso di studi attuale "
                        "(es. liceo, laurea triennale, laurea magistrale, dottorato)."
                    ),
                    "field_of_interest": (
                        "Chiedi allo studente quali materie o aree di studio lo "
                        "interessano di più (es. informatica, medicina, ingegneria, arte)."
                    ),
                    "career_goal": (
                        "Chiedi allo studente cosa vorrebbe fare professionalmente "
                        "dopo la laurea (es. ricerca, industria, insegnamento, pratica clinica)."
                    ),
                    "region": (
                        "Chiedi allo studente in quale area d'Italia vive attualmente "
                        "(es. nord, centro, sud, isole)."
                    ),
                    "mobility_preference": (
                        "Chiedi allo studente se è disposto a trasferirsi per gli studi "
                        "universitari (es. restare in zona, spostarsi, andare ovunque)."
                    ),
                }
                hint = slot_hints[slot]
                return f"[REPHRASE] {hint}"

    # ── RIASEC helpers ───────────────────────────────────────────────
    def _build_riasec_question(self, idx: int, is_first: bool = False, is_retry: bool = False) -> str:
        question = extraction.RIASEC_QUESTIONS[idx]
        options = extraction.LIKERT_OPTIONS_TEXT
        total = len(extraction.RIASEC_QUESTIONS)
        progress = f"[Domanda {idx + 1} / {total}]"

        options_lines = "\n".join(f"- {o.strip()}" for o in options.split(","))

        parts = []
        if is_retry:
            parts.append(
                "Mi dispiace, ma non sono riuscito a registrare la tua risposta "
                "precedente. Potresti rispondere di nuovo scegliendo una delle "
                "cinque opzioni o indicando un numero da 1 a 5?"
            )
        elif is_first:
            parts.append(
                "Grazie! Il tuo profilo è ora completo. "
                f"Ti farò {total} brevi domande sui tuoi interessi per abbinarti "
                "ai programmi universitari più adatti. Rispondi usando una delle "
                "cinque opzioni."
            )

        body = f"{progress} {question}\n\n{options_lines}"
        full_text = ("\n\n".join(parts) + "\n\n" + body) if parts else body

        # Force the exact text through the postprocessor; the Agent's own
        # paraphrase (which is unreliable for non-English text) is discarded.
        self.riasec_response_text = full_text
        self.force_riasec_response = True

        return {body}

    def _handle_riasec_turn(self, utterance: str):
        total = len(extraction.RIASEC_QUESTIONS)

        # ── Goodbye handling (unchanged) ─────────────────────────
        if self.riasec_goodbye_pending:
            if self._is_goodbye(utterance):
                return None
            self.riasec_goodbye_pending = False
        elif self._is_goodbye(utterance):
            self.riasec_goodbye_pending = True
            remaining = total - self.current_riasec_idx
            return (
                f"Capisco che potresti voler concludere la conversazione. "
                f"Ti restano ancora {remaining} domanda/e sugli interessi "
                "prima che io possa generare raccomandazioni accurate. "
                "Vuoi continuare o preferisci davvero dire arrivederci?"
            )

        # ── Interpret the answer to the *currently displayed* question ─
        idx = self.current_riasec_idx
        print(f"[riasec] Answer for Q{idx + 1}/{total}: {utterance!r}")

        valence = extraction.interpret_riasec_answer(idx, utterance)
        print(f"[riasec] Q{idx + 1}/{total} valence = {valence}")

        if valence is None:
            # Do NOT advance. Re-ask the SAME question with the retry preface.
            return self._build_riasec_question(idx, is_retry=True)

        # Record the answer, advance.
        self.profile_state.riasec.valences[idx] = valence
        self.current_riasec_idx = idx + 1
        print(f"[riasec] Recorded. Progress = {self.current_riasec_idx}/{total}")

        if self.current_riasec_idx >= total:
            return self._finalize_riasec_and_recommend()

        return self._build_riasec_question(self.current_riasec_idx, is_first=False)

    def _finalize_riasec_and_recommend(self):
        """Compute scores + profile, persist on profile_state, run the RAG."""
        riasec = self.profile_state.riasec

        scores = extraction.compute_riasec_scores(riasec.valences)
        profile_code, description, confidence = extraction.compute_riasec_profile(scores)

        riasec.scores = scores
        riasec.profile = profile_code
        riasec.description = description
        riasec.confidence = confidence
        riasec.completed = True

        print("\n[riasec] valences:", riasec.valences)
        print("[riasec] scores:", scores)
        print(f"[riasec] profile={profile_code} | confidence={confidence:.3f}")
        print(f"[riasec] description: {description}\n")

        self.riasec_completed = True
        return self._generate_recommendations()

    def _is_goodbye(self, utterance: str) -> bool:
        if not utterance:
            return False
        text = utterance.casefold().strip()
        if len(text) > 40:
            return False
        return bool(
            re.search(
                r"\b(bye|goodbye|quit|exit|stop|" r"ciao|arrivederci|addio|basta|esci|chiudi|termina)\b",
                text,
            )
        )

    def _is_recommendation_followup(self, utterance: str) -> bool:
        text = utterance.casefold()
        phrases = [
            "waiting for your recommendations",
            "waiting for recommendations",
            "where are the recommendations",
            "where are my recommendations",
            "show me the recommendations",
            "show me the recommendation",
            "give me the recommendations",
            "give me the recommendation",
            "recommendations please",
            "still waiting",
            "i am waiting",
            "i'm waiting",
            "hurry up",
            # Italian
            "aspetto le raccomandazioni",
            "aspetto le tue raccomandazioni",
            "dove sono le raccomandazioni",
            "dove sono i consigli",
            "mostrami le raccomandazioni",
            "mostrami i consigli",
            "dammi le raccomandazioni",
            "dammi i consigli",
            "raccomandazioni per favore",
            "consigli per favore",
            "sto aspettando",
            "aspetto ancora",
            "ancora in attesa",
        ]
        return any(phrase in text for phrase in phrases)

    def _generate_recommendation_followup_answer(self, utterance: str) -> str:
        """
        Answer a follow-up question about the already-delivered
        recommendations, in particular capability/suitability questions
        ("Am I really capable of studying X?", "Do I have the right
        background?", "Is this too hard for me?").

        The answer is grounded in:
          - the collected student profile,
          - the RIASEC profile,
          - the previously generated per-programme reasoning
            (`why_it_matches`) in `self.rag_result["validated_departments"]`.

        The method must not invent admission requirements, curricula,
        workloads, or career outcomes that are not present in the
        supplied context.
        """
        profile = self.profile_state.profile
        riasec = self.profile_state.riasec

        riasec_view = None
        if riasec and getattr(riasec, "completed", False):
            riasec_view = {
                "profile": riasec.profile,
                "description": riasec.description,
                "scores": riasec.scores,
                "confidence": riasec.confidence,
            }

        validated = (self.rag_result or {}).get("validated_departments", []) or []
        recommendations_view = [
            {
                "university": rec.get("university"),
                "course": rec.get("course"),
                "course_code": rec.get("course_code"),
                "degree": rec.get("degree"),
                "region": rec.get("region"),
                "why_it_matches": rec.get("why_it_matches"),
            }
            for rec in validated
        ]

        recent_context = self.dialogue_tracker.get_recent_context(max_turns=6)

        prompt = f"""Rispondi sempre e solo in italiano, indipendentemente dalla lingua del testo seguente.

Sei un consulente universitario che risponde a una domanda di approfondimento
su raccomandazioni già fornite.

Il tuo compito:

1. Se lo studente chiede se è personalmente capace / preparato / adatto ai
   programmi consigliati, rispondi in modo diretto e onesto. Collega
   esplicitamente la tua risposta a:
   - il suo background accademico,
   - il suo campo di interesse,
   - il suo obiettivo professionale,
   - il suo profilo di interessi RIASEC (se presente),
   - il ragionamento precedente "why_it_matches" per ogni programma.
2. Se lo studente pone una domanda fattuale su uno dei programmi consigliati
   (livello della laurea, regione, contenuto, ecc.), rispondi solo usando
   quanto documentato nel ragionamento "why_it_matches".
3. Se la domanda richiede informazioni non presenti nel contesto fornito
   (requisiti specifici di ammissione, prerequisiti matematici esatti,
   carico di lavoro, classifiche, tassi di occupazione, borse di studio,
   scadenze, ecc.), dillo chiaramente e consiglia di consultare la pagina
   ufficiale delle ammissioni del programma o di contattare un tutor
   accademico.
4. Se la domanda è fuori tema, dillo con cortesia e offriti di continuare
   a parlare dei programmi consigliati.

Regole rigide:

- Non inventare requisiti di ammissione, esami, voti, curricula, carichi di
  lavoro, sbocchi professionali, borse di studio o classifiche.
- Non promettere l'ammissione. Non garantire il successo.
- Non usare il nome o il genere dello studente come fattore di idoneità.
- Non inventare nuovi programmi o università.
- Mantieni la risposta tra 3 e 6 frasi e rivolgiti allo studente con "tu".
- Non ripetere l'intero elenco delle raccomandazioni parola per parola.

PROFILO DELLO STUDENTE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

PROFILO DI INTERESSI RIASEC:
{json.dumps(riasec_view, ensure_ascii=False, indent=2) if riasec_view else "(nessun dato RIASEC)"}

PROGRAMMI CONSIGLIATI (con ragionamento fondato precedente):
{json.dumps(recommendations_view, ensure_ascii=False, indent=2)}

DIALOGO RECENTE (il più recente in fondo):
{recent_context}

DOMANDA DELLO STUDENTE:
"{utterance}"

Restituisci SOLO il testo della risposta, senza premesse né markdown.
"""

        try:
            answer_llm = llm.get_llm()
            response = answer_llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = str(text).strip()
            if not text:
                raise ValueError("The follow-up model returned an empty response.")
            return text
        except Exception as exc:
            print("[followup] Generation error:", exc)
            return (
                "Non ho informazioni specifiche sufficienti nei dati recuperati "
                "sui programmi per rispondere con precisione. Ti consiglio di "
                "consultare i requisiti di ammissione ufficiali dei programmi "
                "sopra indicati e, se possibile, di parlare con un tutor accademico."
            )

    def _generate_recommendations(self):
        """
        -> Run the recommendation pipeline exactly once when possible.
        - On success, the grounded recommendation text is stored so it can
        be returned verbatim on the current or a later recommendation request.
        - On failure, profile collection is not restarted. A subsequent
        recommendation request can retry the RAG pipeline.
        """

        try:
            rag_result = self.recommend()
            self.rag_result = rag_result  # keep it for follow-up Q&A
            recommendation_text = str(rag_result.get("recommendations", "")).strip()

            if not recommendation_text:
                recommendation_text = (
                    "Mi dispiace, ma non sono riuscito a generare le "
                    "raccomandazioni universitarie dai documenti disponibili."
                )

            self.recommendation_text = recommendation_text
            self.recommendation_ready = True
            self.force_recommendation_response = True
            self.finished = True
            self.post_recommendation = True

            return recommendation_text

        except Exception as exc:
            print("Recommendation generation error:", exc)

            self.recommendation_text = (
                "Mi dispiace, al momento non riesco a recuperare le "
                "raccomandazioni universitarie. Ti chiedo di richiederle "
                "di nuovo."
            )
            self.recommendation_ready = False
            self.force_recommendation_response = True

            # profile_complete remains True, so the next request can retry
            # the recommendation pipeline instead of asking profile questions again
            self.finished = False

            return self.recommendation_text

    def instruct(self, dialog, utterance):

        if utterance is None or utterance.strip() == "":
            if not self.started:
                self.started = True
                self.profile_state.reset()

                return (
                    "Ciao! Aiuto gli studenti a scegliere il percorso universitario "
                    "in base al loro background, ai loro interessi, obiettivi e vincoli.\n\n"
                    "Qual è il tuo attuale percorso di studi e quali aree di studio "
                    "ti interessano di più?"
                )

            return None

        # Profile is already complete and recommendations succeeded
        # if self.finished:

        #     if self._is_recommendation_followup(utterance):
        #         self.force_recommendation_response = True

        #     return None
        # ── Post-recommendation phase ─────────────────────────────────
        if self.post_recommendation:
            text = (utterance or "").casefold().strip()

            closing_markers = [
                "bye",
                "goodbye",
                "quit",
                "exit",
                "stop",
                "thank you",
                "thanks",
                "thankyou",
                "ok thank you",
                "ok, thank you",
                "that's all",
                "that is all",
                "no thanks",
                "no, thanks",
                "no thank you",
                "no, thank you",
                "all good",
                "i'm good",
                "im good",
                # Italian
                "ciao",
                "arrivederci",
                "addio",
                "basta",
                "esci",
                "chiudi",
                "grazie",
                "va bene grazie",
                "ok grazie",
                "tutto bene",
                "è tutto",
                "e tutto",
                "non grazie",
                "no grazie",
                "sono a posto",
            ]
            if any(marker in text for marker in closing_markers):
                self.post_recommendation = False
                return (
                    "Prego, e in bocca al lupo per i tuoi studi! "
                    "Se in futuro avrai bisogno di altro, scrivimi pure. "
                    "Arrivederci!"
                )

            if self._is_recommendation_followup(utterance):
                self.force_recommendation_response = True
                return None

            answer = self._generate_recommendation_followup_answer(utterance)
            self.followup_response_text = answer
            self.force_followup_response = True
            return answer

        # ── Legacy finished branch (kept for safety) ───────────────────
        if self.finished:
            if self._is_recommendation_followup(utterance):
                self.force_recommendation_response = True
            return None

        # ── RIASEC phase ────────────────────────────────────────────
        # Profile is complete but RIASEC is still running.
        if self.profile_complete and not self.riasec_completed:
            return self._handle_riasec_turn(utterance)

        # Now Profile and RIASEC are both complete. Reaching this branch means the previous `_generate_recommendations()`
        # call failed, so we retry the RAG without re-asking any question.
        if self.profile_complete and self.riasec_completed:
            if self._is_goodbye(utterance):
                return None
            return self._generate_recommendations()

        #  profile collection
        extraction.update_profile_from_text(
            utterance, profile_state=self.profile_state, dialogue_tracker=self.dialogue_tracker
        )

        evaluation = self.evaluate_profile_completeness()
        if evaluation["complete"]:
            # Infer gender from the user's name (unchanged).
            extraction.complete_gender_slot(self.profile_state)

            # Profile complete → enter the RIASEC phase instead of the RAG.
            self.profile_complete = True
            self.current_riasec_idx = 0

            return self._build_riasec_question(idx=0, is_first=True)

        # Ask for confirmation if a filled slot has insufficient confidence.
        low_conf_slot = self.lowest_confidence_filled_slot()
        if low_conf_slot is not None and self.profile_state.profile.get(low_conf_slot) is not None:
            pretty = low_conf_slot.replace("_", " ")

            return (
                f"Just to confirm, I understood your {pretty} as "
                f"'{self.profile_state.profile[low_conf_slot]}'. Is that correct?"
            )

        missing_slots = evaluation["missing_slots"]
        if missing_slots:
            return self.generate_question(missing_slots)

        return None


class SecureLengthOrchestrator(LengthOrchestrator):
    def instruct(self, dialog, utterance):
        if dialog is None:
            return None
        return super().instruct(dialog, utterance)


# Module-level holder so the postprocessor can access the orchestrator
# created later inside `__main__`.
global _profile_guard_ref


def postprocess_counselor_response(output_text: str) -> str:
    """
    Ensure that the two LLM-generated, grounded texts are returned verbatim:

      1. the recommendation list, and
      2. the recommendation follow-up answer.

    sdialog calls this with the Agent LLM output string. In both cases the
    orchestrator has already produced the final user-visible text, so the
    Agent must not paraphrase, summarise, or modify it.
    """
    guard = _profile_guard_ref
    if guard is None:
        return output_text

    if guard.force_recommendation_response and guard.recommendation_text:
        guard.force_recommendation_response = False
        return guard.recommendation_text

    if guard.force_followup_response and guard.followup_response_text:
        guard.force_followup_response = False
        return guard.followup_response_text

    if guard.force_riasec_response and guard.riasec_response_text:
        guard.force_riasec_response = False
        return guard.riasec_response_text

    return output_text


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

    import threading

    def _warmup_models():
        try:
            print("[warmup] Loading GLiNER...")
            extraction.get_extractor().model  # triggers lazy load in extraction.py
            print("[warmup] Loading retriever models...")
            retriever = recommendations.get_retriever()  # triggers lazy construction in rag/recommendations.py
            _ = retriever.embedding_model  # triggers lazy load in rag/retrieval.py
            _ = retriever.reranker  # triggers lazy load in rag/retrieval.py
            print("[warmup] All models ready.")
        except Exception as exc:
            # never let a warm-up failure kill the app — the models will simply load on first use instead.
            print("[warmup] Warm-up failed:", exc)

    threading.Thread(target=_warmup_models, daemon=True).start()

    main_llm = llm.get_llm()

    dialogue_tracker = models.DialogueTurnsTracker()

    ood_guard = OutOfDomainGuardOrchestrator(dialogue_tracker, model=main_llm)

    profile_guard = ProfileCollectionOrchestrator(dialogue_tracker)
    _profile_guard_ref = profile_guard
    # length orchestrator
    length_guard = SecureLengthOrchestrator(min=5, max=20)

    _farewell_re = re.compile(
        r"\b(bye|goodbye|quit|exit|ciao|arrivederci|addio|basta|esci|chiudi)\b",
        re.IGNORECASE,
    )
    finish_reflex = SimpleReflexOrchestrator(
        condition=lambda utt: bool(_farewell_re.search(utt or "")),
        instruction=(
            "Grazie per aver utilizzato il servizio di orientamento universitario. "
            "Spero di esserti stato utile. Arrivederci!"
        ),
    )

    # edw ein o man - wx aman
    counselor = Agent(
        persona=Persona(
            name="Consulente Universitario",
            role=(
                "Sei un consulente di orientamento universitario. "
                "Rispondi sempre e solo in italiano, indipendentemente dalla lingua "
                "in cui ti rivolgi l'utente o in cui sono scritti i prompt di sistema.\n\n"
                "L'applicazione controlla la raccolta del profilo, la valutazione "
                "degli interessi RIASEC e le raccomandazioni universitarie tramite "
                "orchestratori deterministici.\n\n"
                "Dopo che lo studente ha completato il profilo, gli farai 8 brevi "
                "domande di autoriflessione sui suoi interessi. Ogni domanda usa una "
                "scala Likert a 5 punti (Fortemente in disaccordo, In disaccordo, "
                "Neutro, D'accordo, Fortemente d'accordo). Le risposte servono a "
                "calcolare un profilo di interessi RIASEC (Realistico, Investigativo, "
                "Artistico, Sociale, Intraprendente, Convenzionale), che informa le "
                "raccomandazioni universitarie.\n\n"
                "Quando ricevi un'istruzione che inizia con '[REPHRASE]', riformulala "
                "in una domanda naturale e amichevole senza cambiarne il significato. "
                "Quando ti viene data una domanda RIASEC, mantieni invariati il testo "
                "della domanda e le cinque opzioni Likert.\n\n"
                "Non fare domande diverse da quelle che ti vengono indicate.\n\n"
                "Non inventare nomi di università, programmi, lauree, regioni o "
                "risultati di raccomandazione.\n\n"
            ),
        ),
        name="Counselor",
        think=False,
        response_details=(
            "Sii conciso, professionale e poni una sola domanda alla volta in base "
            "all'ordine degli slot, agli slot mancanti e all'indice corrente delle "
            "domande RIASEC. Rispondi sempre in italiano."
        ),
        tools=[get_counseling_policy],
        model=main_llm,
        postprocess_fn=postprocess_counselor_response,
    )

    # have the counselor agent pass through the OOD guard, profile guard, length guard, and finish reflex orchestrators
    # counselor_agent acts as the orchestratos of the orchsetrators
    counselor = counselor | ood_guard | profile_guard | length_guard | finish_reflex

    print("\n\nUniversity Career Counselor is running...\n")

    boot = profile_guard.get_welcome_message()
    if boot:
        dialogue_tracker.add_system_turn(boot)
        print("Counselor:", boot)

    exit_words = [
        "quit",
        "exit",
        "bye bye",
        "bye",
        "goodbye",
        "thank you",
        "thanks",
        "ciao",
        "arrivederci",
        "addio",
        "basta",
        "esci",
        "chiudi",
        "grazie",
        "grazie mille",
    ]
    while True:
        user_text = input("User: ")
        dialogue_tracker.add_user_turn(user_text)
        normalized = re.sub(r"[^a-zA-ZÀ-ÿ ]", "", user_text.casefold()).strip()
        if normalized in exit_words:
            break
        reply = counselor(user_text)
        dialogue_tracker.add_system_turn(reply)
        print("Counselor:", reply)
