import config
import numpy as np
import re
import json
import hashlib
import glob
import pickle
import os
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import models
import rag.documents as documents


def build_index_signature() -> str:
    signature_config = {
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "data_dir": str(config.DATA_DIR),
        "index_pipeline_version": config.INDEX_PIPELINE_VERSION,
    }
    payload = json.dumps(signature_config, sort_keys=True).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:16]


def normalize_embeddings(vectors: np.ndarray):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)

    return vectors / norms


class UniversityIndexBuilder:

    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device="cuda")

    # Chunk documents
    def chunk_documents(self, docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
        """
        Split source documents into logical section records and then
        chunk each record while preserving its metadata.

        The source files use '-----' separators to delimit records.
        Each record may represent a section of the same university
        programme, possibly with a different curriculum or URL.
        -> Returns:
            List[models.DocumentChunk]
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
        for source, document_text in docs:

            # Split the merged source file into logical records.
            # For the RAG files, these are primarily separated by '-----'
            records = documents.split_document_records(source=source, text=document_text)

            for record_number, (record_source, record_text) in enumerate(records):

                if not record_text.strip():
                    continue

                # Extract metadata from THIS record.
                metadata = documents.extract_record_metadata(source=record_source, record_text=record_text)

                # Split only this record into chunks.
                split_texts = splitter.split_text(record_text)
                for chunk_number, chunk_text in enumerate(split_texts):

                    if not chunk_text.strip():
                        continue

                    chunks.append(
                        models.DocumentChunk(
                            text=chunk_text,
                            source=source,
                            chunk_id=f"{record_number}_{chunk_number}",
                            university=metadata.get("university"),
                            course=metadata.get("course"),
                            course_code=metadata.get("course_code"),
                            degree=metadata.get("degree"),
                            type=metadata.get("type"),
                            curriculum=metadata.get("curriculum"),
                            section=metadata.get("section"),
                            url=metadata.get("url"),
                            region=metadata.get("region"),
                            explicit_region=metadata.get("explicit_region"),
                        )
                    )

        return chunks

    def build_index(self):
        # load from disk if available, otherwise build from scratch
        cache_path = os.path.join(config.INDEX_CACHE_DIR, f"index_{build_index_signature()}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.documents = data["documents"]
            self.index = data["index"]
            self.region_indices = data["region_indices"]
            print("Loaded FAISS index from cache.")
            return

        raw_docs = documents.load_documents()
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
        for region in config.VALID_REGIONS:

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
