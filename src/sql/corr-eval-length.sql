SELECT
  LEN(review_text) AS review_length,
  evaluation
FROM
  delta_scan('./export/main/');
