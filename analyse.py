import duckdb
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from deltalake import DeltaTable

review_table = "delta_scan('./export/delta-table/')"
distinct_reviews = f"(SELECT DISTINCT(review_id) FROM {review_table})"

duckdb.sql(
    f"SELECT COUNT(CASE WHEN LOWER(main_category) = 'all beauty' THEN 1 END) as beauty_count, COUNT(CASE WHEN LOWER(main_category) = 'amazon fashion' THEN 1 END) as amazon_count FROM {review_table};"
)

# Count total no. of records.
duckdb.sql(f"SELECT COUNT(*) FROM {review_table};")

# Check for duplicates.
duckdb.sql(
    f"SELECT COUNT(*) FROM {review_table} WHERE review_id IN {distinct_reviews};"
)

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
        x=df_one["score"], y=df_one["percentage"], mode="lines", name="1* Reviews"
    )
)
fig.add_trace(
    go.Scatter(
        x=df_five["score"], y=df_five["percentage"], mode="lines", name="5* Reviews"
    )
)
fig.update_layout(template="plotly_dark")
fig.show()

df = duckdb.sql(
    f"WITH score_by_rating AS (SELECT rating, score, COUNT(score) FROM {review_table} WHERE review_id IN (SELECT DISTINCT(review_id) FROM {review_table}) GROUP BY score, rating ORDER BY rating, score) SELECT * FROM score_by_rating WHERE ABS(rating) = 5;"
).pl()
fig = px.line(
    df,
    x="score",
    y="count(score)",
    text="count(score)",
    template="plotly_dark",
    title="LLM Rated Review Quality (5* Reviews)",
)
fig.show()

duckdb.sql(
    f"WITH score_by_rating AS (SELECT rating, score, COUNT(score) as score_count FROM {review_table} WHERE review_id IN (SELECT DISTINCT(review_id) FROM {review_table}) GROUP BY score, rating ORDER BY rating, score) SELECT CASE WHEN score >= 8 THEN SUM(score_count) END, SUM(score_count) FROM score_by_rating WHERE ABS(rating) = 1 GROUP BY score;"
)

# Percentage of 10s by rating.
df = duckdb.sql(
    f"SELECT rating, (COUNT(CASE WHEN score >= 9 THEN 1 ELSE NULL END) / COUNT(score)) * 100 AS ten_percentage FROM {review_table} GROUP BY rating;"
).pl()
fig = px.bar(
    df,
    x="rating",
    y="ten_percentage",
    text="ten_percentage",
    template="plotly_dark",
    title="% of High Quality Reviews (>=9/10) by Rating",
)
fig.show()

duckdb.sql(
    f"WITH score_by_rating AS (SELECT rating, score, COUNT(score) FROM {review_table} WHERE review_id IN (SELECT DISTINCT(review_id) FROM {review_table}) GROUP BY score, rating ORDER BY rating, score) SELECT * FROM score_by_rating WHERE ABS(rating) = 1;"
)

# duckdb.sql(
#     f"SELECT (COUNT(CASE WHEN rating IN (5, 4, 3) THEN 1 END) / COUNT(rating) * 100) AS rating_percentage  FROM {review_table};"
# )
#
#
# duckdb.sql(f"SELECT review_title, review_text FROM {review_table} WHERE score = 1;")
