SELECT
  YEAR(TO_TIMESTAMP(timestamp / 1000)) AS year,
  AVG(evaluation) AS average_evaluation,
  COUNT(*) AS n_evaluations
FROM
  delta_scan('./export/main/')
GROUP BY
  year
ORDER BY
  year;
