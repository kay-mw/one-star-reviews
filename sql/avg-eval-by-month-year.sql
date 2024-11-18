WITH date_buckets AS (
  SELECT
    DATE_TRUNC('month', TO_TIMESTAMP(timestamp / 1000)) AS date_bucket,
    evaluation
  FROM
    delta_scan('./export/main/')
)

SELECT
  AVG(evaluation) AS average_evaluation,
  COUNT(*) AS n_evaluations,
  date_bucket
FROM
  date_buckets
GROUP BY
  date_bucket
ORDER BY
  date_bucket;
