import os
import traceback

os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/app/.cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/app/.cache/huggingface/transformers")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/app/.cache/huggingface/sentence_transformers")
# only enable these after the models have been downloaded at least once.
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
        "description": "The student's gender, only if explicitly stated.",
        "required": False,
    },
    "name": {
        "description": "The student's first name, only if explicitly stated.",
        "required": True,
    },
    "region": {
        "description": (
            "The student's Italian home macro-region. "
            "Infer the macro-region from an explicitly mentioned city, "
            "province, or area when possible. "
            "Canonical values are: north, center, south, islands."
        ),
        "required": True,
    },
    "field_of_interest": {
        "description": ("The academic subject or university study area the student " "wants to study."),
        "required": True,
    },
    "academic_background": {
        "description": (
            "The student's current or completed educational level, "
            "such as high school, bachelor's degree, master's degree, "
            "engineering degree, or PhD."
        ),
        "required": True,
    },
    "mobility_preference": {
        "description": (
            "The student's willingness to relocate for university. Use strict_local, nearby_ok, or italy_ok."
        ),
        "required": True,
    },
    "career_goal": {
        "description": ("The professional or academic goal after graduation."),
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
    """Return the counseling policy, slot order, and required profile fields."""
    return {
        "goal": "Collect student profile and recommend suitable university departments.",
        "required_slots": REQUIRED_SLOTS,
        "slot_order": SLOT_ORDER,
        "style": "Ask one concise question at a time.",
        "rag": "Use retrieved university information to generate grounded recommendations.",
    }


#####################################################################


class OutOfDomainGuardOrchestrator(BaseOrchestrator):

    def __init__(self, dialogue_tracker, model=None):
        super().__init__()

        self.dialogue_tracker = dialogue_tracker
        self.model = model

        self.judgeyesno = LLMJudgeYesNo(
            """
Is the user's latest request unrelated to:
- university studies
- degree selection
- academic interests
- educational background
- university mobility
- career counseling
- region of current living in Italy

Consider carefully what the system previously asked:
{{last_system}}

Here is the user's latest utterance:
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
                return "Politely explain that you only assist with university and career counseling."
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

    def reset(self):
        super().reset()

        self.finished = False
        self.post_recommendation = False
        self.started = False
        self._boot_message_sent = False

        self.profile_complete = False
        self.recommendation_ready = False
        self.recommendation_text = None
        self.force_recommendation_response = False

        self.profile_state.reset()

    def get_welcome_message(self):
        if not self._boot_message_sent:
            self._boot_message_sent = True
            self.started = True
            self.profile_state.reset()

            return (
                "Hello! I help students choose university programs based on "
                "their background, interests, goals and constraints.\n\n"
                "Would you introduce yourself by telling me your first name "
                "and what is your current educational background?"
            )

        return None

    def evaluate_profile_completeness(self):
        return check_profile_completeness(self.profile_state)

    def recommend(self):
        return recommendations.retrieve_universities(self.profile_state.profile)

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

    def _is_goodbye(self, utterance: str) -> bool:
        text = utterance.casefold().strip()

        return any(
            phrase in text
            for phrase in [
                "bye",
                "goodbye",
                "quit",
                "exit",
                "stop",
                "end the conversation",
            ]
        )

    def _is_recommendation_followup(self, utterance: str) -> bool:
        """
        Detect requests for recommendations after profile collection
        without requiring another LLM call.
        """
        text = utterance.casefold()

        phrases = [
            "waiting for your recommendations",
            "waiting for recommendations",
            "where are the recommendations",
            "where are my recommendations",
            "still waiting",
            "i am waiting",
            "i'm waiting",
            "hurry up",
            "recommendations please",
            "show me the recommendations",
            "give me the recommendations",
            "recommendations",
            "recommendation",
        ]

        return any(phrase in text for phrase in phrases)

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
            recommendation_text = str(rag_result.get("recommendations", "")).strip()

            if not recommendation_text:
                recommendation_text = (
                    "I'm sorry, but I could not generate the university "
                    "recommendations from the available documents."
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
                "I'm sorry, but I am currently unable to retrieve the "
                "university recommendations. Please ask for the recommendations "
                "again."
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
                    "Hello! I help students choose university programs based "
                    "on their background, interests, goals and constraints.\n\n"
                    "What is your current educational background and which "
                    "study areas interest you most?"
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
            # 1. goodbye / acknowledgement → close politely
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
            ]
            if any(marker in text for marker in closing_markers):
                self.post_recommendation = False
                return "You're welcome, and good luck with your studies! If you need anything else in the future, feel free to ask. Goodbye!"

            # 2. Follow-up questions about the recommendations → repeat them verbatim
            if self._is_recommendation_followup(utterance):
                self.force_recommendation_response = True
                return None

            # 3. Any other utterance → polite pivot
            return (
                "I'm glad I could help with your university recommendations. "
                "Is there anything else you'd like to ask about them? "
                "Otherwise, just say 'goodbye' and we'll wrap up."
            )

        # ── Legacy finished branch (kept for safety) ───────────────────
        if self.finished:
            if self._is_recommendation_followup(utterance):
                self.force_recommendation_response = True
            return None

        # profile is complete -> generate recommendations
        if self.profile_complete:
            if self._is_goodbye(utterance):
                return None

            return self._generate_recommendations()

        #  profile collection
        extraction.update_profile_from_text(
            utterance, profile_state=self.profile_state, dialogue_tracker=self.dialogue_tracker
        )

        evaluation = self.evaluate_profile_completeness()
        if evaluation["complete"]:
            # Infer gender from user's name
            extraction.complete_gender_slot(self.profile_state)
            # Mark the profile as complete before entering the recommendation phase
            self.profile_complete = True
            return self._generate_recommendations()

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
    Ensure that grounded recommendation output is returned verbatim.

    sdialog calls this with a single argument (the LLM output string).
    The recommendation pipeline already produces the final user-visible
    text. The Counselor Agent must not paraphrase, summarize, or modify it.
    """
    guard = _profile_guard_ref
    if guard is None:
        return output_text

    if guard.force_recommendation_response and guard.recommendation_text:
        guard.force_recommendation_response = False
        return guard.recommendation_text

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

    # finish reflex
    finish_reflex = SimpleReflexOrchestrator(
        condition=lambda utt: any(
            x in utt.lower()
            for x in [
                "bye",
                "goodbye",
                "quit",
                "exit",
                "stop",
                "thank you",
                "thanks",
                "that's all",
                "that is all",
            ]
        ),
        instruction=("Thank you for using the University Career Counseling service. " "I hope I helped you. Goodbye."),
    )

    # edw ein o man - wx aman
    counselor = Agent(
        persona=Persona(
            name="University Career Counselor",
            role=(
                "You are a university career counselor.\n\n"
                "The application controls profile collection and university "
                "recommendations through deterministic orchestrators.\n\n"
                "When you receive an instruction starting with '[REPHRASE]', "
                "rephrase that instruction into a natural, friendly question "
                "without changing its meaning.\n\n"
                "Do NOT ask anything other than the instructed questions you are given.\n\n"
                "Do NOT invent university names, programmes, degrees, regions, "
                "or recommendation results.\n\n"
            ),
        ),
        name="Counselor",
        think=False,
        response_details=(
            "Be concise, professional, and ask only one question at a time based on the given slot order and the missing slots."
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

    while True:
        user_text = input("User: ")
        dialogue_tracker.add_user_turn(user_text)
        if user_text.strip().lower() in ["quit", "exit", "bye bye", "bye", "goodbye", "thank you"]:
            break
        reply = counselor(user_text)
        dialogue_tracker.add_system_turn(reply)
        print("Counselor:", reply)
