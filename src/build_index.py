from pathlib import Path

import config
from rag.indexing import UniversityIndexBuilder, build_index_signature


def main():

    config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = config.INDEX_CACHE_DIR / f"index_{build_index_signature()}.pkl"
    # check if index cache exists to avoid recreating it
    if cache_path.exists():
        print(f"Index cache already exists: {cache_path}")
        print("Skipping index rebuild.")
        return

    builder = UniversityIndexBuilder()
    builder.build_index()

    print("Index build completed.")


if __name__ == "__main__":
    main()

# python build_index.py
