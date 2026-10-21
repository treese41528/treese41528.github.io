-- Slide 13 of the AI at Purdue talk: the TV-news SQL funnel, for the sqlite3 shell.
-- Question: how many ANES TV-news runs passed the AI screen (auditor verdict Compliant or Minor issues),
-- had no extraction warning flagged, and reported support?
-- Data: results.csv from Bertran, Fogliato and Wu's repository at commit d489f03 (SHA-256 b29b60804252c3627bb5394f8f29e870c35c3be06b7594146c079742c95c91f2):
--   https://media.githubusercontent.com/media/amazon-science/agentic-forking-path/d489f03b143fcaa9befb8cc443594ff63ec61eaa/experiment_data/results.csv
-- Run in the folder that holds results.csv:  sqlite3 < tvnews_sql_funnel.sql
--
-- Import schema: .import creates table results from the header row; every column is TEXT, values as released.
-- Flags are the strings 'True' and 'False'. SQLite's TRUE is the integer 1, so hypothesis_supported = TRUE
-- matches no rows; compare with the string 'True'.
-- Empty fields: this shell stores them as ''; extract_bertran_anes.py (Python sqlite3) stores them as NULL.
-- No extraction warning flagged: keep runs whose warning is not 'True'. COALESCE(warning, 'False') treats a missing
-- (NULL) warning as not flagged; this shell stores the blank as '', which is also not 'True'. A bare
-- warning <> 'True' would drop the blank when it is NULL (687 instead of 688).
-- Case: SQLite compares text with its default BINARY collation, so 'compliant' would match no rows here.
.mode csv
.import results.csv results
.mode list
.headers on
-- The query on slide 13 (expected: 122)
SELECT COUNT(*) AS reported_support
FROM results
WHERE dataset = 'sample_anes-inconsistent-views'
  AND sonnet_verdict IN ('Compliant', 'Minor issues')
  AND COALESCE(warning, 'False') <> 'True'
  AND hypothesis_supported = 'True';
-- Each stage (expected: 4,194, 1,270, 700, 688, 122)
SELECT 0 AS step, 'Released results' AS stage, COUNT(*) AS rows FROM results
UNION ALL
SELECT 1, 'TV-news dataset', COUNT(*) FROM results
  WHERE dataset = 'sample_anes-inconsistent-views'
UNION ALL
SELECT 2, 'Auditor verdict Compliant or Minor issues', COUNT(*) FROM results
  WHERE dataset = 'sample_anes-inconsistent-views'
    AND sonnet_verdict IN ('Compliant', 'Minor issues')
UNION ALL
SELECT 3, 'No extraction warning flagged', COUNT(*) FROM results
  WHERE dataset = 'sample_anes-inconsistent-views'
    AND sonnet_verdict IN ('Compliant', 'Minor issues')
    AND COALESCE(warning, 'False') <> 'True'
UNION ALL
SELECT 4, 'Reported support', COUNT(*) FROM results
  WHERE dataset = 'sample_anes-inconsistent-views'
    AND sonnet_verdict IN ('Compliant', 'Minor issues')
    AND COALESCE(warning, 'False') <> 'True'
    AND hypothesis_supported = 'True'
ORDER BY step;
-- Warning values among the runs with the chosen verdicts (expected: 1 (blank), 687 False, 12 True)
SELECT CASE WHEN COALESCE(warning, '') = '' THEN '(blank)' ELSE warning END AS warning, COUNT(*) AS rows FROM results
  WHERE dataset = 'sample_anes-inconsistent-views'
    AND sonnet_verdict IN ('Compliant', 'Minor issues')
  GROUP BY 1 ORDER BY 1;
