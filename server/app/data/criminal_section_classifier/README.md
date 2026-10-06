# Criminal section classifier

Logistic-regression head over BGE-M3 query embeddings that predicts the IPC
sections a lay question engages (118 sections plus `NONE`). `CriminalRAGSystem`
adds the predicted sections to the rerank candidate pool.

- `weights.npz`: `W` (labels x 1024), `b`, `labels`.
- `bns_map.json`: IPC section -> BNS section, only where the section texts match
  closely (embedding similarity >= 0.85). Unmapped sections still reach BNS
  through hybrid retrieval.

## Training data and licence

Trained on the real layperson split (`FT-Layman-train`) of **ILSIC**
(Law-AI, EACL 2026; https://github.com/Law-AI/ilsic), which pairs Indian legal
forum questions with the statutes cited in their answers. The ILSIC data is
released under **CC BY-NC-SA 4.0**, so these weights are a derived work under
the same terms: non-commercial use only, with attribution, shared alike.

Selected on `FT-Layman-dev` (C=16): recall@5 0.844, recall@10 0.909 for
questions that cite an IPC section.
