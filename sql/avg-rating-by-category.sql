SELECT
  main_category,
  AVG(rating) AS average_rating,
  COUNT(*) AS n_ratings
FROM
  delta_scan('./export/main/')
GROUP BY
  main_category
ORDER BY
  average_rating DESC;
