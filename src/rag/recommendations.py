import config
import models
import rag.retrieval as retrieval

# import os
# from typing import List
import json
import llm
from json_repair import repair_json

# import rag.documents as documents
from dataclasses import dataclass, field

# essential...
_retriever_instance = None


@dataclass
class ProgrammeCandidate:
    candidate_id: int
    university: str
    course: str
    course_code: str | None
    degree: str | None
    region: str | None
    url: str | None
    source: str
    score: float
    evidence: list[str] = field(default_factory=list)


def get_retriever():

    global _retriever_instance
    if _retriever_instance is None:

        print("Building university retriever...")
        _retriever_instance = retrieval.UniversityRAGRetriever()

    return _retriever_instance


def count_unique_programmes(retrieval_results: list[tuple[models.DocumentChunk, float]]) -> int:
    """Count unique university programmes in retrieved results."""

    programme_keys = set()

    for chunk, _ in retrieval_results:

        university = (chunk.university or "").strip().casefold()
        course_code = (chunk.course_code or "").strip().casefold()
        course = (chunk.course or "").strip().casefold()

        if not university or not course:
            continue

        if course_code:
            key = (university, course_code)
        else:
            key = (university, course)

        programme_keys.add(key)

    return len(programme_keys)


def build_programme_candidates(retrieval_results: list[tuple[models.DocumentChunk, float]]) -> list[ProgrammeCandidate]:
    """
    Convert retrieved DocumentChunks into unique programme candidates.

    The programme identity is determined from authoritative metadata,
    not by extracting names from arbitrary chunk text.
    """

    candidates_by_key = {}

    for chunk, score in retrieval_results:

        university = (chunk.university or "").strip()
        course = (chunk.course or "").strip()

        if not university or not course:
            continue

        university_key = university.casefold()
        course_code_key = (chunk.course_code or "").strip().casefold()
        course_key = course.casefold()

        if course_code_key:
            key = (university_key, course_code_key)
        else:
            key = (university_key, course_key)

        existing = candidates_by_key.get(key)

        if existing is None:

            candidates_by_key[key] = ProgrammeCandidate(
                candidate_id=len(candidates_by_key),
                university=university,
                course=course,
                course_code=chunk.course_code,
                degree=chunk.degree or chunk.type,
                region=chunk.region,
                url=chunk.url,
                source=chunk.source,
                score=float(score),
                evidence=[chunk.text],
            )

        else:

            if float(score) > existing.score:
                existing.score = float(score)

            if not existing.url and chunk.url:
                existing.url = chunk.url

            if chunk.text not in existing.evidence:
                existing.evidence.append(chunk.text)

    candidates = list(candidates_by_key.values())

    candidates.sort(
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    # Reassign stable IDs after sorting.
    for candidate_id, candidate in enumerate(candidates):
        candidate.candidate_id = candidate_id

    return candidates


def retrieve_universities(profile: dict) -> dict:
    """
    Retrieve grounded university programmes and select recommendations.

    The LLM is used only to select candidate IDs and explain why selected
    programmes match the student. University/programme metadata always comes
    directly from the retrieved database candidates.
    """

    retriever = get_retriever()

    # Stage 1: retrieve using the student's geographic preference
    retrieval_results = retriever.retrieve(profile, min_unique_programmes=5)
    geography_relaxed = False
    if count_unique_programmes(retrieval_results) < 3:
        # Stage 2: relax geographic constraint
        retrieval_results = retriever.retrieve(profile, allowed_regions_override=config.VALID_REGIONS)
        geography_relaxed = True

    candidates = build_programme_candidates(retrieval_results)
    if not candidates:
        return {
            "recommendations": (
                "I couldn't find any specific university programmes "
                "matching your profile in the available documents."
            ),
            "retrieval_results": retrieval_results,
            "validated_departments": [],
        }

    # Stage 3: prepare CLOSED candidate list for the LLM
    candidate_payload = []
    for candidate in candidates:

        candidate_payload.append(
            {
                "id": candidate.candidate_id,
                "university": candidate.university,
                "programme": candidate.course,
                "course_code": candidate.course_code,
                "degree": candidate.degree,
                "region": candidate.region,
            }
        )

    prompt = f"""
You are selecting university programmes from a CLOSED DATABASE.

You are NOT searching for universities.

You MUST select programmes ONLY from the candidate IDs provided below.

You MUST NOT:
- invent a university;
- invent a programme;
- rename a university;
- rename a programme;
- introduce a candidate that is not in the list;
- use outside knowledge to introduce additional universities.

Student profile:
{json.dumps(profile, indent=2)}

Available database candidates:
{json.dumps(candidate_payload, indent=2)}

Select at least 3 candidates that best match the student's:
- field of interest;
- academic background;
- career goal.

If fewer than 3 candidates are suitable, select only the suitable candidates.
If none are suitable, return an empty list.

Return ONLY valid JSON in exactly this format:

{{
    "selected_ids": [0, 1, 2]
}}
"""

    # Stage 4: LLM selects IDs only
    advisor_llm = llm.get_llm(format="json")
    raw_response = advisor_llm.invoke(prompt)
    response_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        try:
            repaired = repair_json(response_text)
            parsed = json.loads(repaired)
        except Exception as exc:
            print("Failed to parse recommendation selection:", exc)

            return {
                "recommendations": ("I'm sorry, I couldn't generate precise " "recommendations right now."),
                "retrieval_results": retrieval_results,
                "validated_departments": [],
            }

    selected_ids = parsed.get("selected_ids", [])
    if not isinstance(selected_ids, list):
        selected_ids = []

    # Stage 5: deterministic validation
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    validated_candidates = []
    for candidate_id in selected_ids:

        if not isinstance(candidate_id, int):
            continue

        candidate = candidate_map.get(candidate_id)
        if candidate is None:
            print(f"Rejected invalid recommendation ID: {candidate_id}")
            continue

        validated_candidates.append(candidate)

    # Keep at most three recommendations.
    validated_candidates = validated_candidates[:3]
    if not validated_candidates:

        return {
            "recommendations": (
                "I'm sorry, but I was unable to find suitable " "programmes in the available university documents."
            ),
            "retrieval_results": retrieval_results,
            "validated_departments": [],
        }

    # Stage 6: generate grounded explanations
    final_recommendations = []
    for candidate in validated_candidates:

        evidence = "\n\n".join(candidate.evidence[:3])
        explanation_prompt = f"""
You are a university career counsellor.

Write a SHORT explanation (EXACTLY 3 to 4 sentences) of why the following university programme is relevant to the student.

IMPORTANT:
Use ONLY the supplied student profile and database evidence. The explanation MUST be between 3-4 sentences.

STRICT RULES:
- do not invent facts;
- do not introduce information about the university that is not supplied;
- do not discuss tourism or city attractions;
- do notprovide generic information about Italy;
- do not invent admission requirements;
- do not change the programme name;
- do not change the university name.
- Each sentence must connect a specific profile slot (field of interest, academic background, career goal, region,
  or mobility preference) to this programme.

Student profile:
{json.dumps(profile, indent=2)}

Database programme:
University: {candidate.university}
Programme: {candidate.course}
Course code: {candidate.course_code}
Degree: {candidate.degree}
Region: {candidate.region}

Database evidence:
{evidence}

Write a short explanation focused specifically on why this programme matches the student's academic background, field of interest, and career goal.
"""

        explanation_response = advisor_llm.invoke(explanation_prompt)
        explanation = (
            explanation_response.content if hasattr(explanation_response, "content") else str(explanation_response)
        )

        final_recommendations.append(
            {
                "university": candidate.university,
                "course": candidate.course,
                "course_code": candidate.course_code,
                "degree": candidate.degree,
                "region": candidate.region,
                "url": candidate.url,
                "source": candidate.source,
                # "why_it_matches": explanation.strip(),
            }
        )

    # Stage 7: deterministic final response construction
    lines = []
    if geography_relaxed:
        lines.extend(
            [
                "I found fewer than three suitable programmes "
                "within your preferred geographic area, so I "
                "expanded the search to other parts of Italy.",
            ]
        )

    lines.append("Here are the recommended university programmes:")
    lines.append("")
    for i, recommendation in enumerate(final_recommendations, start=1):

        lines.append(f"{i}. **{recommendation['university']}**")
        lines.append(f"   Programme: {recommendation['course']}")
        if recommendation["degree"]:
            lines.append(f"   Degree: {recommendation['degree']}")

        if recommendation["region"]:
            lines.append(f"   Region: {recommendation['region']}")

        # lines.append(f"   Why it matches: " f"{recommendation['why_it_matches']}")

        if recommendation["url"]:
            lines.append(f"   More information: " f"{recommendation['url']}")

        lines.append("")

    return {
        "recommendations": "\n".join(lines),
        "retrieval_results": retrieval_results,
        "validated_departments": final_recommendations,
    }
