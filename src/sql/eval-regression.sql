SELECT
  evaluation,
  rating,
  CASE
    WHEN verified_purchase = true THEN 1 ElSE 0
  END AS verified_purchase_converted,
  helpful_vote,
  average_rating,
  price
FROM
  delta_scan('./export/main/')
WHERE
  price IS NOT NULL;
