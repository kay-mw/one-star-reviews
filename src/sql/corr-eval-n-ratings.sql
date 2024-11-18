SELECT
  rating_number,
  evaluation
FROM
  delta_scan('./export/main');
