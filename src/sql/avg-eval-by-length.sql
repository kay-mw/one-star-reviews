SELECT
  LEN(review_text) AS review_length,
  AVG(evaluation) AS average_evaluation,
  COUNT(*) AS n
FROM
  delta_scan('./export/main/')
GROUP BY
  review_length
HAVING
  COUNT(*) > 1
ORDER BY
  review_length;
