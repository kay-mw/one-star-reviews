SELECT
  MONTH(TO_TIMESTAMP(timestamp / 1000)) AS month,
  evaluation
FROM
  delta_scan('./export/main/');
