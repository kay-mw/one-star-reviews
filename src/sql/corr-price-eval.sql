SELECT
  price,
  evaluation
FROM
  delta_scan('./export/main/')
WHERE
  price IS NOT NULL;
