SELECT
  CASE
    WHEN verified_purchase = true THEN 1 ELSE 0
  END AS verified_purchase_converted,
  evaluation
FROM
  delta_scan('./export/main/')
