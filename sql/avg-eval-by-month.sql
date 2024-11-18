SELECT
  MONTH(TO_TIMESTAMP(timestamp / 1000)) AS month,
  AVG(evaluation) AS average_evaluation,
  COUNT(*) AS n_evaluations
FROM
  delta_scan('./export/main/')
GROUP BY
  month
ORDER BY
  month;
