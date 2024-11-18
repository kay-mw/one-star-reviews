SELECT
  AVG(evaluation) AS average_evaluation,
  COUNT(*) AS n_evaluations,
  main_category
FROM
  delta_scan('./export/main/')
GROUP BY
  main_category
ORDER BY
  average_evaluation DESC;
