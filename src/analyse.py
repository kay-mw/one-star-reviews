from typing import List, Literal

import duckdb
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import statsmodels.api as sm

review_table = "delta_scan('./export/delta-table/')"
polars_table = "delta_scan('./export/polars-delta/')"
main_table = "delta_scan('./export/main/')"

distinct_reviews = f"(SELECT DISTINCT(review_id) FROM {review_table})"


def open_sql(name: str) -> str:
    with open(f"./sql/{name}.sql", "r") as file:
        return file.read()


# NOTE: Exploratory analysis
with pl.Config(tbl_cols=-1, tbl_rows=5):
    print(duckdb.sql(open_sql("select-star")).pl())


duckdb.sql(open_sql("stddev-eval-by-rating")).pl()

pl.Config(tbl_rows=-1, set_fmt_str_lengths=10000)

test = (
    duckdb.sql(
        f"""SELECT review_title, review_text, timestamp, product_title, main_category  FROM {main_table} WHERE evaluation = 8 AND timestamp = 1421512926000;"""
    )
    .pl()
    .to_dicts()
)

print(test)

duckdb.sql(open_sql("avg-rating-by-month")).pl()

duckdb.sql(open_sql("avg-rating-by-year")).pl()

duckdb.sql(open_sql("avg-rating-by-month-year")).pl()

duckdb.sql(open_sql("avg-rating-by-category")).pl()

duckdb.sql(open_sql("avg-rating-by-purchase")).pl()

duckdb.sql(open_sql("avg-eval-by-month")).pl()

duckdb.sql(open_sql("avg-eval-by-year")).pl()

duckdb.sql(open_sql("avg-eval-by-month-year")).pl()

duckdb.sql(open_sql("avg-eval-by-category")).pl()

# NOTE: Regression
regression_df = duckdb.sql(open_sql("eval-regression")).pl()
X = regression_df.select(
    "review_length",
    "rating",
    "verified_purchase_converted",
    "price",
    "helpful_vote",
).to_numpy()
y = regression_df.select("evaluation").to_numpy()
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

print(model.summary().as_html())


# NOTE: Correlations
def calc_correlation(
    name: str, col_x: str, col_y: str, method: Literal["pearson"] | Literal["spearman"]
) -> float:
    result = (
        duckdb.sql(open_sql(name)).pl().select(pl.corr(col_x, col_y, method=method)),
    )
    result = result[0][0].item(0, 0)
    return result


eval_length_spearman = calc_correlation(
    name="corr-eval-length",
    col_x="review_length",
    col_y="evaluation",
    method="spearman",
)
eval_length_pearson = calc_correlation(
    name="corr-eval-length",
    col_x="review_length",
    col_y="evaluation",
    method="pearson",
)

eval_rating_spearman = calc_correlation(
    name="corr-eval-rating", col_x="rating", col_y="evaluation", method="spearman"
)
eval_rating_pearson = calc_correlation(
    name="corr-eval-rating", col_x="rating", col_y="evaluation", method="pearson"
)


eval_price_spearman = calc_correlation(
    name="corr-price-eval",
    col_x="price",
    col_y="evaluation",
    method="spearman",
)
eval_price_pearson = calc_correlation(
    name="corr-price-eval",
    col_x="price",
    col_y="evaluation",
    method="pearson",
)


eval_help_spearman = calc_correlation(
    name="corr-help-eval",
    col_x="helpful_vote",
    col_y="evaluation",
    method="spearman",
)
eval_help_pearson = calc_correlation(
    name="corr-help-eval",
    col_x="helpful_vote",
    col_y="evaluation",
    method="pearson",
)


eval_purchase_spearman = calc_correlation(
    name="corr-eval-purchase",
    col_x="verified_purchase_converted",
    col_y="evaluation",
    method="spearman",
)
eval_purchase_pearson = calc_correlation(
    name="corr-eval-purchase",
    col_x="verified_purchase_converted",
    col_y="evaluation",
    method="pearson",
)


# NOTE: Plots
pio.templates.default = "plotly_dark"
pio.templates["plotly_dark"].layout


def scatter(
    data: pl.DataFrame,
    x: str,
    y: str,
    z: str | None,
    size: str | None,
    label_x: str,
    label_y: str,
    label_z: str | None,
    xrange: List[int | float] | None,
    yrange: List[int | float] | None,
    correlation: float,
    xtick: int | None = None,
    ytick: int | None = None,
    outlier: bool = False,
    outlier_text: str = "",
) -> go.Figure:
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=z,
        size=size,
        trendline="ols",
        labels={
            x: label_x,
            y: label_y,
            z: label_z,
        },
    )
    fig.update_xaxes(dtick=xtick, range=xrange)
    fig.update_yaxes(dtick=ytick, range=yrange)
    fig.update_layout(
        title=f"<b>{label_x} by {label_y}</b>",
        xaxis_title=f"<b>{label_x}</b>",
        yaxis_title=f"<b>{label_y}</b>",
        font=dict(size=15),
        annotations=[
            dict(
                x=1,
                y=1,
                xref="paper",
                yref="paper",
                text=f"<b>Correlation Coefficient: {round(correlation, 2)}</b>",
                showarrow=False,
                align="center",
                font=dict(size=18),
            ),
        ],
    )
    if outlier:
        fig.update_layout(
            annotations=[
                dict(
                    x=1,
                    y=1,
                    xref="paper",
                    yref="paper",
                    text=f"<b>Correlation Coefficient: {round(correlation, 2)}</b>",
                    showarrow=False,
                    align="center",
                    font=dict(size=18),
                ),
                dict(
                    x=1,
                    y=0,
                    xref="paper",
                    yref="paper",
                    text=f"<b>{outlier_text}</b>",
                    showarrow=False,
                    align="center",
                    font=dict(size=18, color="red"),
                ),
            ]
        )
    return fig


avg_eval_by_length = duckdb.sql(open_sql("avg-eval-by-length")).pl()
fig = scatter(
    data=avg_eval_by_length,
    x="review_length",
    y="average_evaluation",
    z="n",
    size=None,
    label_x="Review Length (No. Characters)",
    label_y="Review Quality (Avg.)",
    label_z="No. Reviews",
    xrange=[-50, 2_500],
    yrange=[-0.5, 10],
    xtick=250,
    ytick=1,
    correlation=eval_length_pearson,
    outlier=True,
    outlier_text="Outliers >2500 characters excluded",
)
fig.show()
fig.write_image("../media/avg_eval_by_length.png")

avg_eval_by_rating = duckdb.sql(open_sql("avg-eval-by-rating")).pl()
fig = px.line(
    avg_eval_by_rating,
    x="rating",
    y="average_evaluation",
    error_y="stddev_evaluation",
    markers=True,
    labels={
        "rating": "Star Rating",
        "average_evaluation": "Review Quality (Avg.)",
    },
)
fig.update_xaxes(dtick=1)
fig.update_yaxes(dtick=1, range=[0, 10])
fig.update_layout(
    title="<b>Review Quality (Avg.) by Star Rating</b>",
    xaxis_title="<b>Quality</b>",
    yaxis_title="<b>Percentage</b>",
    font=dict(size=15),
    legend=dict(title="<b>Star Rating</b>"),
    annotations=[
        dict(
            x=1,
            y=1,
            xref="paper",
            yref="paper",
            text=f"<b>Correlation Coefficient: {round(eval_rating_spearman, 2)}</b>",
            showarrow=False,
            align="center",
            font=dict(size=18),
        ),
    ],
)
fig.show()
fig.write_image("../media/avg_eval_by_rating.png")

avg_eval_by_price = duckdb.sql(open_sql("avg-eval-by-price")).pl()
fig = scatter(
    data=avg_eval_by_price,
    x="price",
    y="average_evaluation",
    z="n",
    size=None,
    label_x="Price ($USD)",
    label_y="Review Quality (Avg.)",
    label_z="No. Reviews",
    xrange=[-50, 1000],
    yrange=[-0.5, 10],
    ytick=1,
    correlation=eval_price_pearson,
    outlier=True,
    outlier_text="Outliers >$1000 excluded",
)
fig.show()
fig.write_image("../media/avg_eval_by_price.png")

avg_eval_by_help = duckdb.sql(open_sql("avg-eval-by-helpful")).pl()
fig = scatter(
    data=avg_eval_by_help,
    x="helpful_vote",
    y="average_evaluation",
    z="n",
    size="n",
    label_x="Helpful Votes",
    label_y="Review Quality (Avg.)",
    label_z="No. Reviews",
    xrange=[-30, 400],
    yrange=[-1.5, 10],
    ytick=1,
    correlation=eval_help_pearson,
    outlier=True,
    outlier_text="Outliers >400 helpful votes excluded",
)
fig.show()
fig.write_image("../media/avg_eval_by_help_true.png")


avg_eval_by_help = duckdb.sql(open_sql("avg-eval-by-helpful")).pl()
fig = scatter(
    data=avg_eval_by_help,
    x="helpful_vote",
    y="average_evaluation",
    z="n",
    size=None,
    label_x="Helpful Votes",
    label_y="Review Quality (Avg.)",
    label_z="No. Reviews",
    xrange=[-30, 400],
    yrange=[-1.5, 10],
    ytick=1,
    correlation=eval_help_pearson,
    outlier=True,
    outlier_text="Outliers >400 helpful votes excluded",
)
fig.show()
fig.write_image("../media/avg_eval_by_help.png")


pct_eval_by_rating = duckdb.sql(open_sql("pct-eval-by-rating")).pl()
fig = px.line(
    pct_eval_by_rating,
    x="evaluation",
    y="percentage",
    color="rating",
    markers=True,
    labels={
        "evaluation": "<b>Quality</b>",
        "percentage": "<b>Percentage</b>",
        "rating": "<b>Rating</b>",
    },
)
fig.update_xaxes(dtick=1, range=[-0.5, 10])
fig.update_yaxes(tickformat=".2%", range=[-0.03, 0.35])
fig.update_layout(
    title="<b>Distribution of Review Quality by Star Rating</b>",
    xaxis_title="<b>Quality</b>",
    yaxis_title="<b>Percentage</b>",
    font=dict(size=15),
    legend=dict(title="<b>Star Rating</b>"),
)
fig.show()

fig.write_image("../media/pct_eval_by_rating.png")

pct_eval = duckdb.sql(open_sql("pct-eval")).pl()
fig = px.line(
    pct_eval,
    x="evaluation",
    y="percentage",
    markers=True,
    labels={
        "evaluation": "<b>Quality</b>",
        "percentage": "<b>Percentage</b>",
    },
)
fig.update_xaxes(dtick=1, range=[-0.5, 10])
fig.update_yaxes(tickformat=".2%", range=[-0.03, 0.30])
fig.update_layout(
    title="<b>Distribution of Review Quality</b>",
    xaxis_title="<b>Quality</b>",
    yaxis_title="<b>Percentage</b>",
    font=dict(size=15),
)
fig.show()
fig.write_image("../media/pct_eval.png")


running_total = duckdb.sql(open_sql("running-total-by-month-year")).pl()
running_total = running_total.with_columns(
    pl.col("total_reviews").cast(pl.UInt32), pl.col("date_bucket").cast(pl.String)
)
fig = px.line(
    running_total,
    x="date_bucket",
    y="total_reviews",
)
fig.update_layout(
    title="<b>Total No. of Reviews Over Time</b>",
    xaxis_title="<b>Date</b>",
    yaxis_title="<b>Total Reviews</b>",
    font=dict(size=15),
)
fig.show()
fig.write_image(file="../media/running-total.png")


avg_eval_by_purchase = duckdb.sql(open_sql("avg-eval-by-purchase")).pl()
fig = px.bar(
    avg_eval_by_purchase,
    x="verified_purchase",
    y="average_evaluation",
    color="verified_purchase",
    error_y="stddev_evaluation",
    labels={
        "verified_purchase": "Verified Purchase",
        "average_evaluation": "Review Quality (Avg.)",
    },
)
fig.update_yaxes(dtick=1, range=[0, 10])
fig.update_layout(
    title="<b>Review Quality (Avg.) by Verified Purchase Status</b>",
    xaxis_title="<b>Verified Purchase</b>",
    yaxis_title="<b>Average Quality</b>",
    font=dict(size=15),
    annotations=[
        dict(
            x=1,
            y=1,
            xref="paper",
            yref="paper",
            text=f"<b>Correlation Coefficient: {round(eval_purchase_pearson, 2)}</b>",
            showarrow=False,
            align="center",
            font=dict(size=18),
        )
    ],
)
fig.show()
fig.write_image("../media/avg_eval_by_purchase.png")

avg_eval_by_category = duckdb.sql(open_sql("avg-eval-by-category")).pl()
fig = px.bar(
    avg_eval_by_category,
    y="main_category",
    x="average_evaluation",
    color="n",
    orientation="h",
    labels={
        "main_category": "<b>Main Category</b>",
        "average_evaluation": "<b>Review Quality</b>",
        "n": "<b>No. Reviews</b>",
    },
)
fig.update_xaxes(dtick=1, range=[0, 10])
fig.update_layout(
    title=f"<b>Review Quality by Main Category</b>",
    xaxis_title=f"<b>Review Quality</b>",
    yaxis_title=f"<b>Main Category</b>",
    font=dict(size=7),
)
fig.show()
fig.write_image("../media/avg_eval_by_category.png")

duckdb.sql(f"""
SELECT
    AVG(LEN(review_text)) AS average_price,
    RANK() OVER(ORDER BY average_price DESC),
    main_category
FROM
    {main_table}
WHERE
    price IS NOT NULL
GROUP BY
    main_category
ORDER BY
    average_price DESC;
""").pl()


# TODO: Re-interpret helpful votes (non-significant with length included)
# TODO: Maybe set up Github Pages to embed plots as SVGs?
