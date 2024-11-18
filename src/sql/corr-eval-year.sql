SELECT
  YEAR(TO_TIMESTAMP(timestamp / 1000)) AS year,
  evaluation
FROM
  delta_scan('./export/main/');
