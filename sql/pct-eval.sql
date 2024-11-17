SELECT
  evaluation,
  ROUND(COUNT(*) / (
    SELECT COUNT(*) 
    FROM delta_scan('./export/main/')
  ) * 100, 2) AS percentage
FROM
  delta_scan('./export/main/')
GROUP BY
  evaluation;
