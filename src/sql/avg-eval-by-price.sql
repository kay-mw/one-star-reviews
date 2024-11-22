SELECT 
    ROUND(price / 4) * 4 AS price,
    AVG(evaluation) AS average_evaluation,
    COUNT(*) AS n
FROM 
  delta_scan('./export/main/')
WHERE
  price IS NOT NULL
GROUP BY 
  ROUND(price / 4) * 4
ORDER BY 
  price;
