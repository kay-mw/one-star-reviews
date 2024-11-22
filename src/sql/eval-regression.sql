SELECT
  evaluation,
  rating,
  CASE
    WHEN verified_purchase = true THEN 1 ElSE 0
  END AS verified_purchase_converted,
  helpful_vote,
  price,
  LEN(review_text) AS review_length
FROM
  delta_scan('./export/main/')
WHERE
  price IS NOT NULL;
