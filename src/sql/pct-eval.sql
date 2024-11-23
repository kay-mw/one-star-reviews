SELECT
  evaluation,
  COUNT(*) / (
    SELECT COUNT(*) 
    FROM delta_scan('./export/main/')
  ) AS percentage
FROM
  delta_scan('./export/main/')
WHERE
  evaluation IS NOT NULL
GROUP BY
  evaluation;
