# STAT 418 — Chapter 6 Code (GenAI Studio)

Runnable Python for *Computational Methods in Data Science*, **Chapter 6 — Large
Language Models in Data Science**. The course slides show pseudocode and the
signature SDK calls, then link here for the full, runnable versions.

All code uses Purdue's GenAI Studio through Dr. Reese's wrapper SDK
(`genai-studio-sdk`). RCAC provides API access only — the wrapper provides the
ergonomics — so installing it is the first step.

## Setup

1. **Get an API key** — https://genai.rcac.purdue.edu → Settings → Account → API Keys
2. **Install** (a virtual environment is recommended):
   ```bash
   python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Set your key** — the scripts read it from the environment; never hardcode it:
   ```bash
   export GENAI_STUDIO_API_KEY="sk-..."                  # Windows: setx GENAI_STUDIO_API_KEY "sk-..."
   ```
4. **Run any script**:
   ```bash
   python ch6_1_foundations/01_first_chat.py
   ```

> LLM outputs are **stochastic** — your wording will differ from any example in
> the book or slides. That variability is studied formally in §6.8.

> **Rate limit.** GenAI Studio allows roughly **20 requests per minute**. The
> batch-heavy scripts (`ch6_4_annotation/01_annotation_basics.py` and
> `03_quality_control.py`) space their calls ~3.5s apart to stay under it, so a
> full 20-item batch takes about 70 seconds. If you run other scripts (or a
> classmate shares your key) concurrently, you share that budget; throttled
> calls are skipped gracefully rather than crashing.

## Index

| Section | Script | Demonstrates |
|---------|--------|--------------|
| 6.1 Foundations | `ch6_1_foundations/01_first_chat.py` | client setup, list/select models, `chat()`, `chat_complete()` token metadata |
| 6.1 Foundations | `ch6_1_foundations/02_streaming_system_multiturn.py` | `chat_stream()`, model comparison, system messages, `Conversation` multi-turn |
| 6.2 Embeddings | `ch6_2_embeddings/01_generate.py` | `embed()`, batch embedding, `embed_complete()` metadata |
| 6.2 Embeddings | `ch6_2_embeddings/02_similarity_search.py` | cosine similarity, similarity matrix, nearest-neighbor search |
| 6.2 Embeddings | `ch6_2_embeddings/03_what_embeddings_encode.py` | PCA + correlate each PC with known metadata (length / sentiment / category); LDA projection |
| 6.2 Embeddings | `ch6_2_embeddings/04_classification.py` | logistic-regression sentiment classifier on embeddings (+ CV) |
| 6.2 Embeddings | `ch6_2_embeddings/05_embeddings_in_regression.py` | embedding PCs as regression covariates + bootstrap CIs (ties Ch. 4) |
| 6.2 Embeddings | `ch6_2_embeddings/06_concept_arithmetic.py` | "king - man + woman = queen": static (GloVe) vs. contextual embeddings; additive vs. rotation vs. linear-map (needs optional `gensim` — see requirements.txt) |
| 6.3 Preprocessing | `ch6_3_preprocessing/01_token_counting.py` | token heuristics, context-window budget, measured token/word ratio by content type |
| 6.3 Preprocessing | `ch6_3_preprocessing/02_chunking_strategies.py` | fixed / overlap / semantic chunking + embedding comparison of which best preserves meaning |
| 6.3 Preprocessing | `ch6_3_preprocessing/03_preprocessing_pipeline.py` | cleaning (noise vs. signal) + configurable `TextPreprocessor` + academic subclass (pure Python) |
| 6.4 Annotation | `ch6_4_annotation/01_annotation_basics.py` | single-label + structured-JSON + batch annotation (`gemma3:12b`), robust to dropped responses |
| 6.4 Annotation | `ch6_4_annotation/02_annotation_tasks.py` | fine-grained sentiment, named-entity recognition, topic classification |
| 6.4 Annotation | `ch6_4_annotation/03_quality_control.py` | Cohen's kappa, bootstrap CI on agreement, consensus voting (ties Ch. 4) |
| 6.5 RAG | `ch6_5_rag/01_builtin_rag.py` | GenAI Studio knowledge-base API: upload, create KB, link, query with `collections=[kb.id]`; RAG vs. non-RAG |
| 6.5 RAG | `ch6_5_rag/02_manual_rag.py` | RAG from scratch: chunk → embed → retrieve (cosine) → grounded answer; `SimpleRAG`; chunk-size effect |
| 6.5 RAG | `ch6_5_rag/03_rag_evaluation.py` | Precision@k / Recall@k retrieval metrics + LLM faithfulness verification |
| 6.6 Prompting | `ch6_6_prompt_engineering/01_prompts_as_code.py` | versioned templates, delimiters, strict JSON output + parsing |
| 6.6 Prompting | `ch6_6_prompt_engineering/02_few_shot_and_cot.py` | few-shot examples + chain-of-thought (direct vs. step-by-step, zero-shot CoT) |
| 6.6 Prompting | `ch6_6_prompt_engineering/03_self_consistency.py` | majority vote over reasoning paths — the bootstrap analogy (throttled) |
| 6.6 Prompting | `ch6_6_prompt_engineering/04_prompt_iteration_ab.py` | prompt iteration + A/B testing with a paired permutation test (ties Ch. 4; throttled) |
| 6.6 Prompting | `ch6_6_prompt_engineering/05_prompt_patterns.py` | prompt chaining, multi-turn `Conversation`, dataset description, code generation |
| 6.7 Tool Use | `ch6_7_tool_use/01_tool_use.py` | declare a tool with `@tool`, one request→execute→return cycle (`chat_raw`), and the model skipping the tool when it isn't needed (`qwen2.5:72b`) |
| 6.8 Reliability | `ch6_8_reliability/01_consistency.py` | test-retest reliability + paraphrase invariance (throttled) |
| 6.8 Reliability | `ch6_8_reliability/02_calibration.py` | confidence elicitation + reliability diagram + ECE (reproduces book 0.149) |
| 6.8 Reliability | `ch6_8_reliability/03_llm_as_judge.py` | LLM-as-judge rubric scoring (1–5) for tasks without ground truth |
| 6.8 Reliability | `ch6_8_reliability/04_uncertainty.py` | self-consistency uncertainty + bootstrap CI on agreement (ties Ch. 4; throttled) |
| 6.8 Reliability | `ch6_8_reliability/05_evaluation_protocol.py` | full protocol: accuracy/F1/consistency vs. deployment thresholds (throttled, ~2 min) |
| 6.9 Responsible AI | `ch6_9_responsible_ai/01_pii_redaction.py` | PII detection + redaction, `PrivacyAuditor` (pure Python) |
| 6.9 Responsible AI | `ch6_9_responsible_ai/02_bias_detection.py` | differential-treatment probe + response-length bias quantification (throttled) |
| 6.9 Responsible AI | `ch6_9_responsible_ai/03_disclosure_and_governance.py` | AI-use disclosure statements + deployment checklist (pure Python) |
| 6.10 Summary | `ch6_10_summary/01_end_to_end_workflow.py` | end-to-end workflow: preprocess → embed → annotate (self-consistency) → kappa → disclose |

> **§6.5 notes.** `ch6_5_rag/` also contains `rag_core.py` (shared `SimpleRAG`/retrieval helpers imported by scripts 02–03) and `sample_syllabus.md` (the demo document). `01_builtin_rag.py` creates a `STAT418-Ch6.5-Demo` knowledge base on your account and reuses it on re-runs (remove it with `ai.delete_knowledge_base(kb_id)`). Scripts 02–03 use `llama3.2:latest`, not `gemma3`, because retrieval needs an embedding-capable model.

> The embedding scripts select `llama3.2:latest` (3072-d). Similarity/coefficient
> *values* depend on the deployed model version, so your numbers will differ from
> the textbook's — the structure (which texts cluster, which features matter) holds.
