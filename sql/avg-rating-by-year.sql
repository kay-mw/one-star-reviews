SELECT
  YEAR(TO_TIMESTAMP(timestamp / 1000)) AS year,
  AVG(rating) AS average_rating,
  COUNT(*) AS n_ratings
FROM
  delta_scan('./export/main/')
GROUP BY
  year
ORDER BY
  year;
