import asyncio
import gzip
import json
import logging
import os
import shutil
import time

import google.generativeai as genai
import typing_extensions as typing
from dotenv import load_dotenv
from google.api_core import exceptions

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

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


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
    seed: int,
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
                "price": pl.Float32,
                "store": pl.String,
                "categories": pl.List(inner=pl.String),
                "parent_asin": pl.String,
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


class Reviews(typing.TypedDict):
    timestamp: int
    score: int


async def async_analyse_reviews(data: str):
    with open("./prompt.md", "r") as file:
        prompt = file.read() + "\n\n" + data

        model_name = None
        for model_info in genai.list_tuned_models():
            model_name = model_info.name

        assert model_name is not None, "Failed to find any finetuned models."

        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                response_mime_type="text/plain",
                response_schema=list[Reviews],
            ),
        )

    for _ in range(3):
        try:
            response = await model.generate_content_async(prompt)
            line_list = response.text.splitlines()
            valid_lines = [
                line.replace("`", "").strip() for line in line_list if "{" in line
            ]
            valid_string = "".join(valid_lines)
            response = json.loads(valid_string)
            return response
        except json.JSONDecodeError:
            logger.info("Failed to parse response. Skipping...")
            break
        except exceptions.ResourceExhausted:
            n_seconds = 3
            logger.info(f"Rate limited. Waiting {n_seconds}s and retrying...")
            time.sleep(n_seconds)
            continue


open("./prompt_examples.json", "w").close()
[os.remove(f"./review_data/{file}") for file in review_files if file.endswith("jsonl")]
[
    os.remove(f"./product_data/{file}")
    for file in product_files
    if file.endswith("jsonl")
]

for review_file, product_file in zip(review_files, product_files):
    t0 = time.time()
    review_name = review_file.split(".")[0]
    review_path = f"./review_data/{review_name}"

    product_name = product_file.split(".")[0]
    product_path = f"./product_data/{product_name}"

    decompress(review_path)
    decompress(product_path)

    rows = 30
    slice_total = 15
    df = read_data(
        review_path=f"{review_path}.jsonl",
        product_path=f"{product_path}.jsonl",
        slice_init=1000000,
        rows=rows,
        slice_total=slice_total,
        seed=2,
    )

    slices = []
    for i in range(slice_total):
        if i == 0:
            slices.append(df.slice(offset=i, length=rows))
        else:
            slices.append(df.slice(offset=i * rows, length=rows))

    async def main():
        tasks = []
        for sliced_df in slices:
            prompt_dict = sliced_df.select(
                pl.col("review_title"),
                pl.col("review_text"),
                pl.col("timestamp"),
                pl.col("rating"),
                pl.col("product_title"),
            ).to_dicts()
            data = json.dumps(prompt_dict)

            task = asyncio.create_task(async_analyse_reviews(data=data))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        return responses

    responses = asyncio.run(main())

    response_dfs = []
    for response in responses:
        if response is None:
            continue
        else:
            response_dfs.append(
                pl.from_dicts(
                    response, schema={"timestamp": pl.UInt64, "score": pl.UInt8}
                )
            )

    response_df = pl.concat(items=response_dfs)
    final_df = df.join(response_df, on="timestamp", how="inner")

    assert len(final_df) > 0, "No data present in final_df."

    final_df.write_delta(target="./export/polars-delta/", mode="append")

    logger.info(
        f"""Exported final_df. Relevant row counts were...
    df: {len(df)}
    response_df: {len(response_df)}
    final_df: {len(final_df)}"""
    )

    os.remove(f"{review_path}.jsonl")
    os.remove(f"{product_path}.jsonl")

    logger.info(f"Processing {review_name} took {time.time() - t0}s.")
