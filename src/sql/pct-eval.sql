SELECT
  evaluation,
  COUNT(*) / (
    SELECT COUNT(*) 
    FROM delta_scan('./export/main/')
  ) AS percentage
FROM
  delta_scan('./export/main/')
GROUP BY
  evaluation;
