SELECT
  helpful_vote,
  AVG(evaluation) AS average_evaluation,
  COUNT(*) AS n
FROM
  delta_scan('./export/main/')
GROUP BY
  helpful_vote
ORDER BY
  helpful_vote;
