SELECT
  verified_purchase,
  AVG(rating) AS average_rating,
  COUNT(*) AS n_ratings
FROM
  delta_scan('./export/main/')
GROUP BY
  verified_purchase;
