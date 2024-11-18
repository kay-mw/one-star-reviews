WITH date_buckets AS (
  SELECT
    DATE_TRUNC('month', TO_TIMESTAMP(timestamp / 1000)) AS date_bucket,
    rating
  FROM
    delta_scan('./export/main/')
)

SELECT
  AVG(rating) AS average_rating,
  COUNT(*) AS n_ratings,
  date_bucket
FROM
  date_buckets
GROUP BY
  date_bucket
ORDER BY
  date_bucket;
