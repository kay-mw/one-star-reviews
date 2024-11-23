SELECT 
    ROUND(price / 2) * 2 AS price,
    AVG(evaluation) AS average_evaluation,
    COUNT(*) AS n
FROM 
  delta_scan('./export/main/')
WHERE
  price IS NOT NULL
GROUP BY 
  ROUND(price / 2) * 2
ORDER BY 
  price;
