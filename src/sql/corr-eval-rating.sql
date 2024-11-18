SELECT
  rating,
  evaluation
FROM
  delta_scan('./export/main/');
