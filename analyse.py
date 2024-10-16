import duckdb
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from deltalake import DeltaTable
from plotly.subplots import make_subplots

review_table = "delta_scan('./export/delta-table/')"
distinct_reviews = f"(SELECT DISTINCT(review_id) FROM {review_table})"

duckdb.sql(
    f"""CREATE TABLE reviews AS
WITH 
    cte AS (
SELECT 
    review_title, 
    review_text, 
    timestamp, 
    main_category, 
    rating, 
    score, 
    user_id, 
    helpful_vote, 
    verified_purchase, 
    product_title, 
    rating_number, 
    price, 
    store, 
    bought_together,
    TO_TIMESTAMP(timestamp / 1000) AS datetime,
    row_number() OVER(PARTITION BY timestamp ORDER BY timestamp) AS rn
FROM 
    delta_scan('./export/delta-table/')
)
SELECT 
    review_title, 
    review_text, 
    timestamp, 
    main_category, 
    rating, 
    score, 
    user_id, 
    helpful_vote, 
    verified_purchase, 
    product_title, 
    rating_number, 
    price, 
    store, 
    bought_together,
    datetime
FROM 
    cte 
WHERE 
    rn = 1;
"""
)

duckdb.sql("SELECT COUNT(*) FROM reviews;")
duckdb.sql(f"SELECT COUNT(DISTINCT(timestamp)) FROM {review_table};")

score_by_category = duckdb.sql(
    """WITH rows AS (
    SELECT COUNT(*) as n_rows, main_category
    FROM reviews
    GROUP BY main_category
)
SELECT 
    rev.main_category, 
    rev.score, 
    (COUNT(rev.score) / row.n_rows) * 100 as percentage 
FROM 
    reviews AS rev LEFT JOIN rows AS row
ON
    rev.main_category = row.main_category
WHERE 
    rev.score IS NOT NULL AND rev.main_category IS NOT NULL AND row.n_rows > 100
GROUP BY 
    rev.score, 
    rev.main_category,
    row.n_rows
ORDER BY 
    score;"""
).pl()
fig = px.bar(
    score_by_category,
    y="main_category",
    x="percentage",
    color="score",
    template="plotly_dark",
    title="Score by Category",
    orientation="h",
)
fig.update_xaxes(dtick=5)
fig.show()

average_score_by_timestamp = duckdb.sql(
    """
WITH time_buckets AS (
    SELECT 
        DATE_TRUNC('month', TO_TIMESTAMP(timestamp / 1000)) as time_bucket,
        score
    FROM reviews
)
SELECT 
    AVG(score) as average_score, 
    time_bucket
FROM 
    time_buckets
GROUP BY 
    time_bucket
ORDER BY 
    time_bucket;
"""
).pl()
fig = px.line(
    average_score_by_timestamp,
    x="time_bucket",
    y="average_score",
    template="plotly_dark",
    title="Average LLM Score by Timestamp",
)
fig.show()

# stddev_score_by_timestamp = duckdb.sql(
#     """
# WITH time_buckets AS (
#     SELECT
#         DATE_TRUNC('month', TO_TIMESTAMP(timestamp / 1000)) as time_bucket,
#         score
#     FROM reviews
# )
# SELECT
#     STDDEV(score) as average_score,
#     time_bucket
# FROM
#     time_buckets
# GROUP BY
#     time_bucket
# ORDER BY
#     time_bucket;
# """
# ).pl()
# fig = px.line(
#     stddev_score_by_timestamp,
#     x="time_bucket",
#     y="average_score",
#     template="plotly_dark",
#     title="Standard dev. LLM Score by Timestamp",
# )
# fig.show()

count_cumulative_score_by_timestamp = duckdb.sql(
    """
WITH time_buckets AS (
    SELECT 
        DATE_TRUNC('month', TO_TIMESTAMP(timestamp / 1000)) as time_bucket,
        score
    FROM reviews
)
SELECT 
    SUM(COUNT(score)) OVER(ORDER BY time_bucket) as cumulative_reviews,
    time_bucket
FROM 
    time_buckets
GROUP BY 
    time_bucket
ORDER BY 
    time_bucket;
"""
).pl()
count_cumulative_score_by_timestamp = count_cumulative_score_by_timestamp.select(
    pl.col("cumulative_reviews").cast(pl.Int64), pl.col("time_bucket")
)
fig = px.line(
    count_cumulative_score_by_timestamp,
    x="time_bucket",
    y="cumulative_reviews",
    template="plotly_dark",
    title="Cumulative Reviews by Timestamp",
)
fig.show()


count_score_by_timestamp = duckdb.sql(
    """
WITH time_buckets AS (
    SELECT 
        DATE_TRUNC('month', TO_TIMESTAMP(timestamp / 1000)) as time_bucket,
        score
    FROM reviews
)
SELECT 
    COUNT(score) As new_reviews,
    time_bucket
FROM 
    time_buckets
GROUP BY 
    time_bucket
ORDER BY 
    time_bucket;
"""
).pl()
fig = px.bar(
    count_score_by_timestamp,
    x="time_bucket",
    y="new_reviews",
    template="plotly_dark",
    title="Cumulative Reviews by Timestamp",
)
fig.show()


count_rating_by_timestamp = duckdb.sql(
    """
WITH time_buckets AS (
    SELECT 
        DATE_TRUNC('month', TO_TIMESTAMP(timestamp / 1000)) as time_bucket,
        rating
    FROM reviews
)
SELECT 
    AVG(rating) AS average_rating,
    time_bucket
FROM 
    time_buckets
GROUP BY 
    time_bucket
ORDER BY 
    time_bucket;
"""
).pl()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(
        x=average_score_by_timestamp["time_bucket"],
        y=average_score_by_timestamp["average_score"],
        name="Average Score by Month",
    ),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=count_score_by_timestamp["time_bucket"],
        y=count_score_by_timestamp["new_reviews"],
        name="No. of New Reviews by Month",
        mode="lines",
    ),
    secondary_y=True,
)
fig.update_layout(template="plotly_dark")
fig.show()

df = duckdb.sql(
    """
SELECT
    review_title,
    review_text,
    score,
    datetime
FROM reviews
ORDER BY 
    score DESC
LIMIT 
    100;
"""
).pl()
with pl.Config(tbl_rows=-1, set_fmt_str_lengths=4000):
    print(df)


duckdb.sql("DESCRIBE SELECT * FROM reviews;")

# fig = px.line(
#     count_rating_by_timestamp,
#     x="time_bucket",
#     y="average_rating",
#     template="plotly_dark",
#     title="Cumulative Reviews by Timestamp",
# )
# fig.show()

## # # # # # # # # # ## # # # # # # # # # ## # # # # # # # # # ## # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # #


duckdb.sql("""SELECT COUNT(*) FROM reviews WHERE score = 1;""")

duckdb.sql("SELECT * FROM reviews;")

duckdb.sql(f"DESCRIBE SELECT * FROM {review_table};")

# Count total no. of records.
duckdb.sql(f"SELECT COUNT(*) FROM {review_table};")

# Check for duplicates.

duckdb.sql(f"SELECT * FROM {review_table};")

# Plot score distribution.
df = duckdb.sql(
    f"SELECT score, COUNT(score) FROM {review_table} WHERE review_id IN (SELECT DISTINCT(review_id) FROM {review_table}) GROUP BY score ORDER BY score;"
).pl()
fig = px.line(
    df,
    x="score",
    y="count(score)",
    text="count(score)",
    template="plotly_dark",
    title="LLM Rated Review Quality (Overall)",
)
fig.show()

# Plot rating distribution.
df = duckdb.sql(
    f"SELECT rating, COUNT(rating) FROM {review_table} WHERE review_id IN (SELECT DISTINCT(review_id) FROM {review_table}) GROUP BY rating ORDER BY rating;"
).pl()
fig = px.line(
    df,
    x="rating",
    y="count(rating)",
    text="count(rating)",
    template="plotly_dark",
    title="Amazon Rating Distribution (Overall)",
)
fig.show()

# Standard deviation
df = duckdb.sql(
    f"SELECT rating, STDDEV(score) FROM {review_table} GROUP BY rating ORDER BY rating;"
).pl()
fig = px.line(
    df,
    x="rating",
    y="stddev(score)",
    text="stddev(score)",
    template="plotly_dark",
    title="Standard Deviation by Rating",
)
fig.show()

# Plot score distribution for 1 and 5 star reviews.
fig = go.Figure()
df_one = duckdb.sql(
    f"""WITH score_by_rating AS (
    SELECT rating, 
        score, 
        COUNT(score) AS score_count 
    FROM 
        {review_table} 
    WHERE 
        review_id IN (SELECT DISTINCT(review_id) FROM {review_table}) 
    GROUP BY 
        score, 
        rating 
    ORDER BY 
        rating, 
        score
    )
    SELECT 
        score, 
        score_count / (
            SELECT SUM(score_count) 
            FROM score_by_rating 
            WHERE rating = 1
        ) * 100 AS percentage
    FROM 
        score_by_rating 
    WHERE 
        rating = 1 
    ORDER BY 
        score;"""
).pl()
df_five = duckdb.sql(
    f"""WITH score_by_rating AS (
    SELECT rating, 
        score, 
        COUNT(score) AS score_count 
    FROM 
        {review_table} 
    WHERE 
        review_id IN (SELECT DISTINCT(review_id) FROM {review_table}) 
    GROUP BY 
        score, 
        rating 
    ORDER BY 
        rating, 
        score
    )
    SELECT 
        score, 
        score_count / (
            SELECT SUM(score_count) 
            FROM score_by_rating 
            WHERE rating = 5
        ) * 100 AS percentage 
    FROM 
        score_by_rating 
    WHERE 
        rating = 5
    ORDER BY 
        score;"""
).pl()
fig.add_trace(
    go.Scatter(
        x=df_one["score"],
        y=df_one["percentage"],
        mode="lines",
        name="1* Reviews",
    )
)
fig.add_trace(
    go.Scatter(
        x=df_five["score"],
        y=df_five["percentage"],
        mode="lines",
        name="5* Reviews",
    )
)
fig.update_layout(template="plotly_dark")

duckdb.sql(
    f"""WITH n_by_category AS (
SELECT COUNT(*) as n_rows, main_category FROM {review_table} GROUP BY main_category
),
tens_by_category AS (
SELECT COUNT(CASE WHEN ABS(score) = 10 THEN 1 END) as n_tens,
main_category
FROM {review_table}
GROUP BY main_category
)
SELECT n.main_category, (tens.n_tens / n.n_rows) * 100 as pct
FROM tens_by_category AS tens LEFT JOIN n_by_category as n
ON tens.main_category = n.main_category
ORDER BY pct DESC;
"""
)

duckdb.sql(
    f"""SELECT ROUND(STDDEV(score), 2) stddev, main_category
FROM {review_table}
GROUP BY main_category
ORDER BY stddev DESC;
"""
)
