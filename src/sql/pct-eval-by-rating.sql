WITH data AS (
  SELECT
    evaluation,
    rating,
    COUNT(*) AS count_per_group
  FROM
    delta_scan('./export/main/')
  WHERE
    evaluation IS NOT NULL
  GROUP BY
    evaluation, rating
),
rating_totals AS (
  SELECT
    rating,
    SUM(count_per_group) AS total_for_rating
  FROM
    data
  GROUP BY
    rating
)
SELECT
  d.evaluation,
  d.rating,
  d.count_per_group / rt.total_for_rating AS percentage
FROM
  data d
JOIN
  rating_totals rt
ON
  d.rating = rt.rating
ORDER BY
  d.rating, d.evaluation;
