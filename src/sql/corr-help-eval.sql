SELECT
  helpful_vote,
  evaluation
FROM
  delta_scan('./export/main/');
