from typing import Literal

import duckdb
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
    "rating",
    "verified_purchase_converted",
    "helpful_vote",
    # "average_rating",
    "price",
).to_numpy()
y = regression_df.select("evaluation").to_numpy()
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# NOTE: Correlations
def calc_correlation(
    name: str, col_x: str, col_y: str, method: Literal["pearson"] | Literal["spearman"]
) -> float:
    result = (
        duckdb.sql(open_sql(name)).pl().select(pl.corr(col_x, col_y, method=method)),
    )
    result = result[0][0].item(0, 0)
    return result


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

pct_eval_by_rating = duckdb.sql(open_sql("pct-eval-by-rating")).pl()
fig = go.Figure()
for rating in sorted(
    pct_eval_by_rating.select("rating").unique().to_series().to_list()
):
    filtered_data = pct_eval_by_rating.filter(pl.col("rating") == rating)
    fig.add_trace(
        go.Scatter(
            x=filtered_data["evaluation"],
            y=filtered_data["percentage"],
            line=dict(
                width=3,
            ),
            name=str(rating),
        )
    )
fig.update_xaxes(dtick=1, range=[0, 10])
fig.update_yaxes(tickformat=".2%")
fig.update_layout(
    title="<b>Distribution of Review Quality by Star Rating</b>",
    xaxis_title="<b>Quality</b>",
    yaxis_title="<b>Percentage</b>",
    font=dict(size=15),
    legend=dict(title="<b>Star Rating</b>"),
)
fig.show()
fig.write_image("../media/pct-eval-by-rating.png")


avg_eval_by_rating = duckdb.sql(open_sql("avg-eval-by-rating")).pl()
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=avg_eval_by_rating["rating"],
        y=avg_eval_by_rating["average_evaluation"],
        error_y=dict(
            type="data",
            array=avg_eval_by_rating["stddev_evaluation"],
            color="rgba(99, 110, 250, 0.5)",
        ),
        line=dict(
            width=3,
        ),
    )
)
fig.update_yaxes(dtick=1, range=[0, 10])
fig.update_xaxes(dtick=1)
fig.update_layout(
    title="<b>Review Quality (Avg.) by Review Star Rating</b>",
    xaxis_title="<b>Rating</b>",
    yaxis_title="<b>Average Quality</b>",
    font=dict(size=15),
    annotations=[
        dict(
            x=1,
            y=1,
            xref="paper",
            yref="paper",
            text=f"<b>Spearman's R: {round(eval_rating_spearman, 2)}</b>",
            showarrow=False,
            align="center",
            font=dict(size=18),
        )
    ],
)
fig.show()
fig.write_image("../media/avg_eval_by_rating.png")


avg_eval_by_purchase = duckdb.sql(open_sql("avg-eval-by-purchase")).pl()
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=avg_eval_by_purchase["verified_purchase"],
        y=avg_eval_by_purchase["average_evaluation"],
        error_y=dict(
            type="data",
            array=avg_eval_by_purchase["stddev_evaluation"],
        ),
    )
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
            text=f"<b>Spearman's R: {round(eval_purchase_spearman, 2)}</b>",
            showarrow=False,
            align="center",
            font=dict(size=18),
        )
    ],
)
fig.show()
fig.write_image("../media/avg_eval_by_purchase.png")


avg_eval_by_price = duckdb.sql(open_sql("avg-eval-by-price")).pl()
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=avg_eval_by_price["price_category"],
        y=avg_eval_by_price["average_evaluation"],
        error_y=dict(
            type="data",
            array=avg_eval_by_price["stddev_evaluation"],
            color="rgba(99, 110, 250, 0.5)",
        ),
        line=dict(
            width=3,
        ),
    )
)
fig.update_yaxes(dtick=1, range=[0, 10])
fig.update_layout(
    title="<b>Review Quality (Avg.) by Product Price Range</b>",
    xaxis_title="<b>Price ($USD)</b>",
    yaxis_title="<b>Average Quality</b>",
    font=dict(size=15),
    annotations=[
        dict(
            x=1,
            y=1,
            xref="paper",
            yref="paper",
            text=f"<b>Spearman's R: {round(eval_price_spearman, 2)}</b>",
            showarrow=False,
            align="center",
            font=dict(size=18),
        )
    ],
)
fig.show()
fig.write_image("../media/avg_eval_by_price.png")


avg_eval_by_help = duckdb.sql(open_sql("avg-eval-by-helpful")).pl()
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=avg_eval_by_help["helpful_vote_range"],
        y=avg_eval_by_help["average_evaluation"],
        error_y=dict(
            type="data",
            array=avg_eval_by_help["stddev_evaluation"],
            color="rgba(99, 110, 250, 0.5)",
        ),
        line=dict(
            width=3,
        ),
    )
)
fig.update_yaxes(dtick=1, range=[0, 10])
fig.update_layout(
    title="<b>Review Quality (Avg.) by Review Helpful Votes</b>",
    xaxis_title="<b>Helpful Votes</b>",
    yaxis_title="<b>Average Quality</b>",
    font=dict(size=15),
    annotations=[
        dict(
            x=1,
            y=1,
            xref="paper",
            yref="paper",
            text=f"<b>Spearman's R: {round(eval_help_spearman, 2)}</b>",
            showarrow=False,
            align="center",
            font=dict(size=18),
        )
    ],
)
fig.show()
fig.write_image("../media/avg_eval_by_help.png")

pct_eval = duckdb.sql(open_sql("pct-eval")).pl()
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=pct_eval["evaluation"],
        y=pct_eval["percentage"],
        line=dict(
            width=3,
        ),
    )
)
fig.update_xaxes(dtick=1, range=[0, 10])
fig.update_yaxes(tickformat=".2%")
fig.update_layout(
    title="<b>Distribution of Review Quality</b>",
    xaxis_title="<b>Quality</b>",
    yaxis_title="<b>Percentage</b>",
    font=dict(size=15),
)
fig.show()
fig.write_image("../media/pct-eval.png")

running_total = duckdb.sql(open_sql("running-total-by-month-year")).pl()
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=running_total["date_bucket"].cast(pl.String),
        y=running_total["total_reviews"].cast(pl.UInt32),
        line=dict(
            width=3,
        ),
    )
)
fig.update_layout(
    title="<b>Total No. of Reviews Over Time</b>",
    xaxis_title="<b>Date</b>",
    yaxis_title="<b>Total Reviews</b>",
    font=dict(size=15),
)
fig.show()
fig.write_image(file="../media/running-total.png")

# TODO: Create plotting function to reduce repetition.
