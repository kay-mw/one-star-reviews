SELECT
  MONTH(TO_TIMESTAMP(timestamp / 1000)) AS month,
  AVG(rating) AS average_rating,
  COUNT(*) AS n_ratings
FROM
  delta_scan('./export/main/')
GROUP BY
  month
ORDER BY
  month;
