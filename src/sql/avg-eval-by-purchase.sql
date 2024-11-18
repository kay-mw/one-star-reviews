SELECT
  verified_purchase,
  AVG(evaluation) AS average_evaluation,
  COUNT(*) AS n_evaluations,
  STDDEV(evaluation) AS stddev_evaluation
FROM
  delta_scan('./export/main/')
GROUP BY
  verified_purchase
ORDER BY
  verified_purchase DESC;
