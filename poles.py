# TODO: When labelling the outputted JSON data, first send it to an LLM to get the basic response structure and score ballparks.
# You can then change score values as you see fit!

import gzip
import json
import logging
import os
import shutil
import time

os.environ["POLARS_MAX_THREADS"] = "6"
import polars as pl

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="main.log",
    encoding="utf-8",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s: %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
)

review_files = sorted(os.listdir("./review_data/"))
product_files = sorted(os.listdir("./product_data/"))


def decompress(file_path_no_ext: str) -> None:
    with gzip.open(f"{file_path_no_ext}.jsonl.gz", "rb") as f_in:
        with open(f"{file_path_no_ext}.jsonl", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def read_data(
    review_path: str,
    product_path: str,
    slice_init: int,
    rows: int,
    slice_total: int,
    seed=1,
) -> pl.DataFrame:
    logger.info(f"Processing {review_path}, {product_path}...")
    logger.info(f"Seed is {seed}.")

    lazy_review = (
        pl.scan_ndjson(
            source=review_path,
            schema={
                "rating": pl.Float32,
                "title": pl.String,
                "text": pl.String,
                # "images": pl.List(
                #     inner=pl.Struct(
                #         {
                #             "small_image_url": pl.String,
                #             "medium_image_url": pl.String,
                #             "large_image_url": pl.String,
                #             "attachment_type": pl.String,
                #         }
                #     )
                # ),
                "asin": pl.String,
                "parent_asin": pl.String,
                "user_id": pl.String,
                "timestamp": pl.UInt64,
                "verified_purchase": pl.Boolean,
                "helpful_vote": pl.UInt32,
            },
        )
        .slice(offset=0, length=slice_init)
        .rename({"title": "review_title", "text": "review_text"})
    )

    lazy_product = (
        pl.scan_ndjson(
            source=product_path,
            schema={
                "main_category": pl.String,
                "title": pl.String,
                "average_rating": pl.Float32,
                "rating_number": pl.Int32,
                # "features": pl.List(inner=pl.String),
                # "description": pl.List(inner=pl.String),
                "price": pl.Float32,
                # "images": pl.List(
                #     inner=pl.Struct(
                #         {
                #             "thumb": pl.String,
                #             "large": pl.String,
                #             "variant": pl.String,
                #             "hi_res": pl.Boolean,
                #         }
                #     )
                # ),
                # "videos": pl.List(
                #     inner=pl.Struct(
                #         {"title": pl.String, "url": pl.String, "user_id": pl.String}
                #     )
                # ),
                "store": pl.String,
                "categories": pl.List(inner=pl.String),
                # "details": pl.DataType,
                "parent_asin": pl.String,
                # "bought_together": pl.List(inner=pl.String),
            },
        )
        .slice(offset=0, length=slice_init)
        .rename({"title": "product_title"})
    )

    lazy_master = lazy_review.join(lazy_product, on="parent_asin", how="inner")
    sample_df = lazy_master.collect(streaming=True).sample(
        n=rows * slice_total, seed=seed
    )

    return sample_df


open("./prompt_examples.json", "w").close()
[os.remove(f"./review_data/{file}") for file in review_files if file.endswith("jsonl")]
[
    os.remove(f"./product_data/{file}")
    for file in product_files
    if file.endswith("jsonl")
]

for review_f, product_f in zip(review_files, product_files):
    t0 = time.time()
    review_name = review_f.split(".")[0]
    review_p = f"./review_data/{review_name}"

    product_name = product_f.split(".")[0]
    product_p = f"./product_data/{product_name}"

    decompress(review_p)
    decompress(product_p)

    df = read_data(
        review_path=f"{review_p}.jsonl",
        product_path=f"{product_p}.jsonl",
        slice_init=1000000,
        rows=10,
        slice_total=1,
    )

    df_dict = df.select(
        pl.col("review_title"),
        pl.col("review_text"),
        pl.col("timestamp"),
        pl.col("rating"),
        pl.col("product_title"),
    ).to_dicts()
    df_str = json.dumps(df_dict)

    with open("./prompt_examples.json", "a") as file:
        file.write(df_str + "\n\n")

    os.remove(f"{review_p}.jsonl")
    os.remove(f"{product_p}.jsonl")

    logger.info(f"Processing {review_name} took {time.time() - t0}s.")
