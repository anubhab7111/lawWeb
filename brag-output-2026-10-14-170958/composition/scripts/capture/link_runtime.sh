#!/usr/bin/env bash
# Give a lawWeb worktree the gitignored runtime files: .env, index binaries, fastText model.
set -e
WT=$1/server
MAIN=/home/ushtro/Documents/dev/lawWeb/server
IL=/home/ushtro/Documents/dev/lawWeb/.claude/worktrees/iltur-corpus-v2/server/app/data/faiss_index
cp "$MAIN/.env" "$WT/.env"
mkdir -p "$WT/app/data/models"
ln -sfn "$MAIN/app/data/models/lid.176.bin" "$WT/app/data/models/lid.176.bin"
for f in index.faiss index.pkl meta.pkl; do
  cp "$MAIN/app/data/faiss_index/unified/$f" "$WT/app/data/faiss_index/unified/"
  cp "$MAIN/app/data/faiss_index/case_law/$f" "$WT/app/data/faiss_index/case_law/"
done
for d in judgments precedent statute_classifier; do
  ln -sfn "$IL/$d" "$WT/app/data/faiss_index/$d"
done
echo "linked runtime into $WT"
