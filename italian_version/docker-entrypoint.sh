#!/bin/sh
set -e

echo "**** Italian University Career Counselor App ****"

echo " Building/Loading RAG cache... "

python -u /app/src/build_index.py

echo " **** Starting interactive application... **** "

exec python -u /app/src/university_counselor.py
