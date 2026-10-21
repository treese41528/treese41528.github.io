Reproducibility files for slide 13 of "AI at Purdue" (October 21, 2026)

codebook_retrieval.py            Ranks the soccer-study codebook's 31 passages by cosine similarity to a question
                                 (all-MiniLM-L6-v2, revision 1110a24). A local demonstration, separate from
                                 Purdue GenAI Studio. Usage and environment are in the script's docstring.
soccer_codebook_retrieval.json   Its output: every passage's similarity and rank; the top 3 are marked as
                                 selected candidate context. Similarity is not the probability of answering.
tvnews_sql_funnel.sql            The slide's SQL count for the sqlite3 shell, with the import schema, the
                                 inclusion policy for the warning flag, and the expected counts.
bertran2026_sql_count.json       The same counts from Python's sqlite3 (empty fields as NULL).

Inputs (pinned to commit d489f03 of github.com/amazon-science/agentic-forking-path):
  codebook  https://raw.githubusercontent.com/amazon-science/agentic-forking-path/d489f03b143fcaa9befb8cc443594ff63ec61eaa/data/soccer/codebook.txt
            SHA-256 926e289e2f563e61b27464725a40d1932fda76099fd699a33cb2d75875af4ca6
  results   https://media.githubusercontent.com/media/amazon-science/agentic-forking-path/d489f03b143fcaa9befb8cc443594ff63ec61eaa/experiment_data/results.csv
            SHA-256 b29b60804252c3627bb5394f8f29e870c35c3be06b7594146c079742c95c91f2

The codebook describes the crowdsourced soccer data of Silberzahn et al. (2018); the results are the run-level
release of Bertran, Fogliato and Wu (PNAS, 2026). Text quoted from the codebook in the JSON output comes from that
repository, which is distributed under the Apache License 2.0 (LICENSE-agentic-forking-path.txt; its notice is
in NOTICE-agentic-forking-path.txt).
