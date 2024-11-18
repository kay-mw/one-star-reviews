SELECT
  average_rating,
  evaluation
FROM
  delta_scan('./export/main/');
