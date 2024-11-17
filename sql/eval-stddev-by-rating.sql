SELECT
  rating,
  STDDEV(evaluation)
FROM
  delta_scan('./export/main/')
GROUP BY
  rating
ORDER BY
  rating;
