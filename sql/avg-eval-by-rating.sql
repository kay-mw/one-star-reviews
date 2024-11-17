SELECT
  rating,
  AVG(evaluation) AS average_evaluation
FROM
  delta_scan('./export/main/')
GROUP BY
  rating
ORDER BY
  rating;
