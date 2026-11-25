#!/bin/sh
set -e

echo "**** University Career Counselor App ****"

echo " Building/loading RAG cache... "

python -u /app/src/build_index.py

echo " **** Starting interactive application... **** "

exec python -u /app/src/university_counselor.py
