SELECT
  verified_purchase,
  AVG(evaluation) AS average_evaluation,
  COUNT(*) AS n_evaluations
FROM
  delta_scan('./export/main/')
GROUP BY
  verified_purchase;
