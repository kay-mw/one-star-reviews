SELECT
  AVG(evaluation) AS average_evaluation,
  COUNT(*) AS n,
  CASE
    WHEN main_category = 'AMAZON FASHION' THEN 'Amazon Fashion'
    WHEN main_category = 'SUBSCRIPTION BOXES' THEN 'Subscription Boxes'
    ELSE main_category
  END AS main_category
  -- REPLACE(
  --   REPLACE(
  --     REPLACE(
  --       CAST(LIST_TRANSFORM(
  --         STRING_SPLIT(main_category, ' '), 
  --         x -> CONCAT(
  --           UPPER(SUBSTRING(x, 1, 1)),
  --           LOWER(SUBSTRING(x, 2, LEN(x)))
  --         )
  --       ) AS VARCHAR), '[', ''
  --     ), ']', ''
  --   ), ',', ''
  -- ) AS main_category
FROM
  delta_scan('./export/main/')
WHERE
  main_category IS NOT NULL AND main_category != ''
GROUP BY
  main_category
ORDER BY
  average_evaluation;
