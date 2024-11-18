WITH date_buckets AS (
  SELECT
    DATE_TRUNC('month', TO_TIMESTAMP(timestamp / 1000)) AS date_bucket,
    rating
  FROM
    delta_scan('./export/main/')
)

SELECT
  date_bucket,
  SUM(COUNT(*)) OVER(ORDER BY date_bucket) AS total_reviews
FROM
  date_buckets
GROUP BY
  date_bucket;
