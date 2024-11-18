SELECT
  CASE
    WHEN helpful_vote <= 0 THEN '0'
    WHEN helpful_vote BETWEEN 1 AND 2 THEN '1-2'
    WHEN helpful_vote BETWEEN 3 AND 4 THEN '3-4'
    WHEN helpful_vote BETWEEN 5 AND 9 THEN '5-9'
    WHEN helpful_vote BETWEEN 10 AND 19 THEN '10-19'
    WHEN helpful_vote BETWEEN 20 AND 29 THEN '20-29'
    WHEN helpful_vote BETWEEN 30 AND 39 THEN '30-39'
    WHEN helpful_vote BETWEEN 40 AND 49 THEN '40-49'
    WHEN helpful_vote BETWEEN 50 AND 99 THEN '50-99'
    WHEN helpful_vote BETWEEN 100 AND 199 THEN '100-199'
    WHEN helpful_vote BETWEEN 200 AND 299 THEN '200-299'
    WHEN helpful_vote BETWEEN 300 AND 500 THEN '300-399'
    ELSE '400-INFINITY'
  END AS helpful_vote_range,
  AVG(evaluation) AS average_evaluation,
  COUNT(*) AS n_evaluations,
  STDDEV(evaluation) AS stddev_evaluation
FROM
  delta_scan('./export/main/')
GROUP BY
  helpful_vote_range
ORDER BY
  LEN(helpful_vote_range), helpful_vote_range;
