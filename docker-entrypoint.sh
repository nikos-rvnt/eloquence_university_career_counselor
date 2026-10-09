#!/bin/sh
set -e

echo "**** University Career Counselor App ****"


echo ""
echo "Building/loading RAG cache..."
echo ""

python -u /app/src/build_index.py

echo ""
echo "**** Starting interactive application... ****"
echo ""

exec python -u /app/src/university_counselor.py