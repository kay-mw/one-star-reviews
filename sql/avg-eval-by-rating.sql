SELECT
  rating,
  COUNT(*) as n_ratings,
  AVG(evaluation) AS average_evaluation
FROM
  delta_scan('./export/main/')
GROUP BY
  rating
ORDER BY
  rating;
