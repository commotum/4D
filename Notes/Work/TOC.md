# Repository Papers — Table of Contents

This consolidated TOC lists all Markdown papers in the repo, grouped by theme. Each entry shows the paper title and a brief description, with the source path in parentheses.

## CSP‑Rules Core (Engine, Theory, Usage)
- Overview — High‑level architecture, reading order, loaders, apps, and where to look for core pieces; contrasts CSP‑Rules with traditional CSP solvers. (Overview.md)
- Notation — How solver output is printed and how to read chains/whips/braids; symbol keys and app overrides. (Notation.md)
- Model — Mental model and API: templates, typed variables, link construction, phases, salience, and rule families. (Model.md)
- Graphs — Node/edge model for candidates, g‑candidates, links/glinks; how graphs are built and used by pattern rules. (Graphs.md)
- State — Working memory facts, globals, lifecycle, mutation semantics, and observability of the running system. (State.md)
- Trigger — Scope → Trigger → Action taxonomy; pattern families (BRT, Subsets, Chains, ORk) and where they live. (Trigger.md)
- Beyond — How non‑binary constraints are encoded as binary via typed CSP variables and relation links across applications. (Beyond.md)
- T&E — Trial & Error and DFS: child contexts, contradiction detection, forcing T&E, and integration with the same rules. (T&E.md)
- Salience — Scheduling: BRT → init‑links → play; app refinements and how to mirror ordering in vectorized implementations. (salience.md)

## CSP → Vectorized (Tensorized Solver Design)
- Vectorized Foundations — Boolean‑tensor state layout, reductions for Singles/Subsets, chain frontiers, typed/glink variants, and batched hypotheses. (Vectorized.md)
- CSP → Vectorized — Step‑by‑step recipe to map CSP‑Rules semantics to vectorized kernels and shapes across apps. (CSP-to-Vectorized.md)

## Spacetime/ARC‑AGI, Attention, and Embeddings
- My MonSTERs & Me — Position‑as‑structure: four‑dimensional extension of RoPE via Minkowski Space‑Time Embedding Rotors (MonSTER), motivation, scaling, and practical notes. (4 Dimension Attention.md)
- ARC‑AGI Notes — Dataset, curriculum, and a MonSTER‑enabled pipeline for code generation and evaluation on ARC‑style tasks. (ARC.md)
- Sequence‑to‑Sequence Model Limitations (ARC‑AGI) — Critique of seq2seq HRM on ARC; why it misses meta‑learning across examples and how 1‑D RoPE/flattening/padding exacerbate training and compute. (HRM Seq2Seq Limitations on ARC-AGI.md)
- Selective Hearing — Proposal for RL‑style dynamic sparse attention to focus compute on relevant tokens in multi‑grid ARC contexts; pairs with MonSTER for isotropic spatiotemporal encoding. (Selective Hearing.md)
- STA (Spacetime Algebra) Sketch — Notes on using Clifford/STA (Cl(1,3)) rotors (boosts + rotations) to generalize RoPE while preserving the Minkowski metric. (STA.md)
- RoPE Demo – Proof — Checklist and arguments for verifying RoPE properties and blockwise generalization. (v.md)
- Analysis — Review of 4‑D RoPE generalization (MonSTER), Sudoku data/RoPE implementation, and how attention can learn Sudoku structure; dataset insights. (analysis.md)
- My Notes — Draft thoughts on quaternion/capsule representations, spatial biases in transformers, and motivation for 4‑D structural embeddings. (My Notes.md)
- Type & Value Embeddings — Split tokens into (TYPE, VALUE); up‑project minimal value representations (e.g., RGB as quaternions) with real/quaternion heads; direct decoding without |V|-softmax. (type-value-embeddings.md)
- Vectorized MonSTER Triad (NumPy) — Block‑wise 4‑D rotor cache and transform without loops; Minkowski dot and RoPE‑style identity/norm checks. (v12.py)

## EDB + MyCloud (Time‑Travel DB, Schema, Indexing)
- EDB Requirements — Scope, terminology, functional/non‑functional requirements, schema growth‑only, tuples, types, query/pull, storage, sync. (EDB-REQUIREMENTS.md)
- EDB Plan — Consolidated build plan and phases for EDB and MyCloud; decisions, testing strategy, and next actions. (EDB-PLAN.md)
- EDB Research Log — Survey/checklist with summaries, insights, constraints, and actions across repo artifacts and references. (EDB-RESEARCH.md)
- Insights — Core invariants and design tenets distilled from Datomic; what to keep/borrow/evolve for EDB. (insights.md)
- Build Outline — Recommended MVP build order with plain‑language rationale for each step. (build-outline.md)
- Value Types (Scalars) — Canonical scalar set (long/double/string/keyword/uuid/instant/ref/bytes/uint8, etc.), external forms, canonical encodings, ordering. (value-types.md)
- Tuple / Composite Encoding — Canonical, lexicographically sortable tuple value (v1): layout, constraints (homogeneous in MVP), accessors, SQL pushdown. (tuple-encoding.md)
- Indexing Strategy — EAVT/AVET/AEVT/VAET on SQLite/Postgres, time travel, background merging, uniqueness, and tuple handling. (indexing-strategy.md)
- MyCloud Overview — Hero app over EDB: documents/blocks/tags, component modeling, value types/tuples, sync, and UX principles. (MyCloud-Overview.md)
