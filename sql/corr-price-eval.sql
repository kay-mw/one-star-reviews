SELECT
  price,
  evaluation
FROM
  delta_scan('./export/main/')
