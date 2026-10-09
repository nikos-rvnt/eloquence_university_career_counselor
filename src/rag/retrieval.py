import numpy as np
import config
from sentence_transformers import SentenceTransformer, CrossEncoder
import rag.indexing as indexing
import pickle
import rag.documents as documents
import models
from collections import defaultdict

import os

os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/app/.cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/app/.cache/huggingface/transformers")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/app/.cache/huggingface/sentence_transformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def normalize_embeddings(vectors: np.ndarray):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)

    return vectors / norms


def get_allowed_study_regions(home_region: str | None, mobility_preference: str | None) -> set[str] | None:
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


class UniversityRAGRetriever:

    def __init__(self):

        self._embedding_model = None
        self._reranker = None

        # self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device=config.DEVICE)
        # self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=config.DEVICE)

        self.documents = []
        # global FAISS index -> used when the student is allowed to study
        # anywhere in Italy or when no geographic restriction is known
        self.index = None
        # region-specific FAISS indices -> the integer values stored in these indices
        # correspond to positions in self.documents.
        self.region_indices = {}

        # self.build_index()
        self.load_index()

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            print("[retrieval] Loading SentenceTransformer...")
            self._embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device=config.DEVICE)
            print("[retrieval] SentenceTransformer ready.")
        return self._embedding_model

    @property
    def reranker(self):
        if self._reranker is None:
            print("[retrieval] Loading CrossEncoder reranker...")
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=config.DEVICE)
            print("[retrieval] CrossEncoder ready.")
        return self._reranker

    def load_index(self) -> None:
        """
        Load a previously built FAISS index from the cache.
        The cache contains:
            documents
            index
            region_indices
        The global index stores all document chunks.
        Each region-specific index stores:
            - a FAISS index
            - document_indices mapping local FAISS positions
            back to positions in self.documents.
        """

        cache_path = os.path.join(config.INDEX_CACHE_DIR, f"index_{indexing.build_index_signature()}.pkl")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"FAISS index cache not found: {cache_path}. Run build_index.py before starting the application."
            )

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

        except (OSError, pickle.UnpicklingError) as exc:
            raise RuntimeError(f"Failed to load FAISS index cache: {cache_path}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid FAISS index cache format: {cache_path}")

        required_keys = {
            "documents",
            "index",
            "region_indices",
        }

        missing_keys = required_keys.difference(data.keys())
        if missing_keys:
            raise RuntimeError("Invalid FAISS index cache. " f"Missing keys: {sorted(missing_keys)}")

        self.documents = data["documents"]
        self.index = data["index"]
        self.region_indices = data["region_indices"]

        if self.index is None:
            raise RuntimeError("Loaded global FAISS index is None.")

        if not self.documents:
            raise RuntimeError("Loaded FAISS index contains no documents.")

        if self.index.ntotal != len(self.documents):
            raise RuntimeError(
                "FAISS/document mismatch: "
                f"index contains {self.index.ntotal} vectors, "
                f"but {len(self.documents)} documents were loaded."
            )

        print(f"Loaded FAISS index from cache: " f"{cache_path} " f"({len(self.documents)} chunks)")

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

    def _programme_key(self, chunk: models.DocumentChunk) -> tuple[str, str] | None:
        """
        Return the authoritative identity of a university programme.
        - Preferred identity:
            university + course_code
        - Fallback:
            university + course
        """

        university = (getattr(chunk, "university", None) or "").strip().casefold()
        course_code = (getattr(chunk, "course_code", None) or "").strip().casefold()
        course = (getattr(chunk, "course", None) or "").strip().casefold()

        if not university:
            return None
        if course_code:
            return university, course_code
        if course:
            return university, course

        return None

    def _diversify_by_programme(
        self,
        candidates: list[tuple[models.DocumentChunk, float]],
        max_per_programme: int = 3,
    ) -> list[tuple[models.DocumentChunk, float]]:
        """
        Keep only a limited number of chunks from each programme.
        This prevents multiple sections of the same programme from
        consuming the entire retrieval pool.
        """

        counts: dict[tuple[str, str], int] = {}
        diversified = []

        for chunk, score in candidates:

            key = self._programme_key(chunk)
            if key is None:
                continue

            current_count = counts.get(key, 0)
            if current_count >= max_per_programme:
                continue

            counts[key] = current_count + 1
            diversified.append((chunk, score))

        return diversified

    # Natural-language profile query
    def build_query(self, profile):

        parts = []

        gender = profile.get("gender")
        if gender:
            parts.append(f"User's gender: {gender}.")

        field = profile.get("field_of_interest")
        if field:
            parts.append(f"User's field of interest: {field}.")

        background = profile.get("academic_background")
        if background:
            parts.append(f"User's academic background: {background}.")

        target_level = self._target_study_level(background)
        if target_level == "undergraduate":
            parts.append("Target study level: undergraduate Bachelor's degree.")
        elif target_level == "masters":
            parts.append("Target study level: Master's degree.")
        elif target_level == "doctoral":
            parts.append("Target study level: doctoral / PhD.")

        goal = profile.get("career_goal")
        if goal:
            parts.append(f"User's career goal: {goal}.")

        region = profile.get("region")
        if region:
            parts.append(f"User's home region: {region}.")

        mobility = profile.get("mobility_preference")
        if mobility:
            parts.append(f"User's mobility preference: {mobility}.")

        parts.append(
            "Recommend at least 3 suitable university degree programmes and departments "
            "that are relevant to the student's field of interest, academic background, "
            "study level,and career goal."
        )

        return "\n".join(parts)

    def _deduplicate_programmes(
        self, reranked: list[tuple[models.DocumentChunk, float]]
    ) -> list[tuple[models.DocumentChunk, float]]:
        """
        Keep only the highest-scoring chunk for each university programme.
        Programme identity is based on:
            university + course_code
        If no course code is available:
            university + course
        """
        best_by_programme = {}

        for chunk, score in reranked:

            university = (chunk.university or "").strip().casefold()
            course_code = (chunk.course_code or "").strip().casefold()
            course = (chunk.course or "").strip().casefold()

            if not university or not course:
                continue

            if course_code:
                key = (university, course_code)
            else:
                key = (university, course)

            current = best_by_programme.get(key)
            if current is None or float(score) > float(current[1]):
                best_by_programme[key] = (chunk, float(score))

        result = list(best_by_programme.values())
        result.sort(key=lambda item: item[1], reverse=True)

        return result

    def _target_study_level(self, academic_background: str | None) -> str | None:
        """
        Convert the student's educational background into the
        university study level that should be recommended.
        Rules:
            high school -> undergraduate
            bachelor's degree -> master's
            master's degree -> doctoral
            phd -> doctoral
        """
        value = str(academic_background or "").strip().casefold()
        if not value:
            return None

        # High-school level -> Bachelor / undergraduate
        high_school_terms = (
            "high school",
            "scientific high school",
            "classical high school",
            "humanities high school",
            "linguistic high school",
            "art high school",
            "music high school",
            "technical institute",
            "professional institute",
            "secondary school",
            "scuola superiore",
            "liceo",
        )

        if any(term in value for term in high_school_terms):
            return "undergraduate"

        # Bachelor's level -> Master's degree
        bachelor_terms = ("bachelor", "bsc", "laurea triennale", "first cycle")
        if any(term in value for term in bachelor_terms):
            return "masters"

        # Master's level -> doctoral/PhD
        master_terms = ("master", "msc", "laurea magistrale", "second cycle", "engineering degree")
        if any(term in value for term in master_terms):
            return "doctoral"

        if any(term in value for term in ("phd", "doctorate", "dottorato")):
            return "doctoral"

        return None

    def _programme_study_level(self, chunk: models.DocumentChunk) -> str | None:
        """
        Determine the educational level of a programme from its
        authoritative degree/type metadata.
        """
        metadata = " ".join(
            value for value in [getattr(chunk, "degree", None), getattr(chunk, "type", None)] if value
        ).casefold()
        if not metadata:
            return None

        # Exclude postgraduate 'First-Level Master's Degree' and
        # 'Second-Level Master's Degree' from the normal MSc category.
        if (
            "first-level master's degree" in metadata
            or "first level master's degree" in metadata
            or "second-level master's degree" in metadata
            or "second level master's degree" in metadata
        ):
            return "postgraduate_master"

        # Normal academic Master's Degree
        if (
            "master's degree" in metadata
            or "master degree" in metadata
            or "laurea magistrale" in metadata
            or "second cycle" in metadata
        ):
            return "masters"

        # Normal academic Bachelor's / undergraduate degree
        if (
            "bachelor's degree" in metadata
            or "bachelor degree" in metadata
            or "bachelor" in metadata
            or "laurea" in metadata
            or "first cycle" in metadata
        ):
            return "undergraduate"

        # Doctoral
        if "phd" in metadata or "doctorate" in metadata or "doctoral" in metadata or "dottorato" in metadata:
            return "doctoral"

        return None

    def _is_compatible_programme(self, chunk: models.DocumentChunk, target_level: str | None) -> bool:
        """
        Hard educational-level constraint.
        If the student's level cannot be determined, do not impose
        a study-level restriction.
        Otherwise, programme level must match the target level.
        """
        if target_level is None:
            return True

        programme_level = self._programme_study_level(chunk)
        if programme_level is None:
            return False

        return programme_level == target_level

    def retrieve(
        self,
        profile: dict,
        top_k: int = config.TOP_K,
        final_k: int = config.FINAL_K,
        allowed_regions_override: set[str] | None = None,
        min_unique_programmes: int = 3,
    ):
        """
        Retrieve university programmes while enforcing the student's
        educational level.
        Guarantees at least `min_unique_programmes` (3) distinct compatible
        programmes whenever at least that many programmes exist in the
        available geographic scope.
        Retrieval priority:
            1. geographic policy
            2. educational-level compatibility
            3. semantic relevance
            4. programme diversity
        """

        query = self.build_query(profile)
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = normalize_embeddings(query_embedding)

        # Determine target educational level.
        target_level = self._target_study_level(profile.get("academic_background"))

        # Determine geographic constraint.
        if allowed_regions_override is not None:
            allowed_regions = allowed_regions_override
        else:
            allowed_regions = get_allowed_study_regions(
                home_region=profile.get("region"), mobility_preference=profile.get("mobility_preference")
            )

        # Large candidate pool.
        # We retrieve substantially more than 3 programmes because
        # many chunks may belong to the same programme.
        candidate_k = max(top_k * 20, final_k * 50, 300)

        semantic_candidates = self._retrieve_by_regions(
            query_embedding=query_embedding,
            allowed_regions=allowed_regions,
            top_k=candidate_k,
        )
        if not semantic_candidates:
            return []

        # Hard filter: educational compatibility.
        # A high-school student must not receive Master's programmes.
        # A bachelor's graduate must not receive Bachelor's programmes.
        compatible_candidates = []
        for chunk, semantic_score in semantic_candidates:

            if self._is_compatible_programme(chunk, target_level):
                compatible_candidates.append((chunk, semantic_score))

        if not compatible_candidates:
            return []

        # Programme-level diversity before expensive reranking.
        compatible_candidates = self._diversify_by_programme(compatible_candidates, max_per_programme=3)
        if not compatible_candidates:
            return []

        candidates = [chunk for chunk, _ in compatible_candidates]

        # Rerank only educationally compatible programmes.
        rerank_pairs = [(query, chunk.text) for chunk in candidates]
        rerank_scores = self.reranker.predict(rerank_pairs)
        reranked = sorted(zip(candidates, rerank_scores), key=lambda item: float(item[1]), reverse=True)

        # Programme-level deduplication.
        deduped = self._deduplicate_programmes(reranked)
        if len(deduped) >= min_unique_programmes:
            return deduped

        # Not enough programmes under the original geographic
        # constraint.
        # Expand geographically while keeping the educational-level
        # constraint HARD.
        italy_regions = config.VALID_REGIONS
        if allowed_regions != italy_regions:
            expanded_candidates = self._retrieve_by_regions(
                query_embedding=query_embedding, allowed_regions=italy_regions, top_k=candidate_k
            )

            expanded_compatible = []
            for chunk, semantic_score in expanded_candidates:

                if not self._is_compatible_programme(chunk, target_level):
                    continue

                expanded_compatible.append((chunk, semantic_score))

            expanded_compatible = self._diversify_by_programme(expanded_compatible, max_per_programme=3)
            if expanded_compatible:
                expanded_chunks = [chunk for chunk, _ in expanded_compatible]
                expanded_pairs = [(query, chunk.text) for chunk in expanded_chunks]
                expanded_scores = self.reranker.predict(expanded_pairs)

                expanded_reranked = sorted(
                    zip(expanded_chunks, expanded_scores), key=lambda item: float(item[1]), reverse=True
                )

                # Combine the original geographically-preferred
                # programmes with the expanded Italian pool.
                combined = deduped + expanded_reranked
                combined = self._deduplicate_programmes(combined)
                if len(combined) >= min_unique_programmes:
                    return combined

                # Still fewer than 3: return every compatible
                # programme that could be found.
                if combined:
                    return combined

        # If the database genuinely contains fewer than 3 compatible
        # programmes, return all compatible programmes.
        return deduped

    def _select_diverse(
        self,
        reranked: list[tuple[models.DocumentChunk, float]],
        min_unique_programmes: int = 3,
    ) -> list[tuple[models.DocumentChunk, float]]:
        """
        Force at least `min_unique_programmes` distinct entries, promoting
        diversity across universities and programmes.
        Strategy:
        * Pass 1 — one chunk per unique (university, course_code|course).
        * Pass 2 — one chunk per NEW university (any programme).
        * Pass 3 — any remaining chunks in score order.
        """

        selected: list[tuple[models.DocumentChunk, float]] = []
        seen_chunks: set[int] = set()
        seen_universities: set[str] = set()
        seen_programmes: set[tuple[str, str]] = set()

        # ── Pass 1: one chunk per unique programme ──────────────────────
        for chunk, score in reranked:
            uni = (getattr(chunk, "university", "") or "").strip().casefold()
            cc = (getattr(chunk, "course_code", "") or "").strip().casefold()
            co = (getattr(chunk, "course", "") or "").strip().casefold()

            if not uni or not (cc or co):
                continue

            key = (uni, cc) if cc else (uni, co)
            if key in seen_programmes:
                continue

            seen_programmes.add(key)
            seen_universities.add(uni)
            seen_chunks.add(id(chunk))
            selected.append((chunk, score))

            if len(selected) >= min_unique_programmes:
                return selected

        # ── Pass 2: one chunk per NEW university ────────────────────────
        for chunk, score in reranked:
            if id(chunk) in seen_chunks:
                continue

            uni = (getattr(chunk, "university", "") or "").strip().casefold()
            if not uni or uni in seen_universities:
                continue

            seen_universities.add(uni)
            seen_chunks.add(id(chunk))
            selected.append((chunk, score))

            if len(selected) >= min_unique_programmes:
                return selected

        # ── Pass 3: whatever remains, in score order ────────────────────
        for chunk, score in reranked:
            if id(chunk) in seen_chunks:
                continue

            seen_chunks.add(id(chunk))
            selected.append((chunk, score))

            if len(selected) >= min_unique_programmes:
                break

        return selected
