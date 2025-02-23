import asyncio
import gzip
import json
import logging
import os
import shutil
import time
from typing import Any, List

import google.generativeai as genai
import typing_extensions
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

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def decompress(file_path_no_ext: str) -> None:
    with gzip.open(f"{file_path_no_ext}.jsonl.gz", "rb") as f_in:
        with open(f"{file_path_no_ext}.jsonl", "wb") as f_out:
            shutil.copyfileobj(fsrc=f_in, fdst=f_out)


def read_data(
    review_path: str,
    product_path: str,
    slice_init: int,
    rows: int,
    slice_total: int,
) -> pl.DataFrame:
    logger.info(f"Processing {review_path}, {product_path}...")

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
    sample_df = lazy_master.collect(streaming=True).sample(n=rows * slice_total)

    return sample_df


class Reviews(typing_extensions.TypedDict):
    timestamp: int
    score: int


def get_model(name: str) -> str:
    model_name = None
    for model_info in genai.list_tuned_models():
        model_name = model_info.name
        if name in model_name:
            return model_name
        else:
            continue

    assert model_name is not None, f"Failed to find model {name}."

    return model_name


async def async_analyse_reviews(data: str) -> List[dict] | None:
    with open("./prompt.md", "r") as file:
        prompt = file.read() + "\n\n" + data

        model_name = get_model(name="geminiflashtuneordered-s9fulltbj4")

        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                response_mime_type="text/plain",
                response_schema=list[Reviews],
                # max_output_tokens=2000,
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
            print(response.text)
            break
        except exceptions.ResourceExhausted:
            n_seconds = 3
            logger.info(f"Rate limited. Waiting {n_seconds}s and retrying...")
            time.sleep(n_seconds)
            continue
        except ValueError:
            logger.info(f"Prompt was blocked. Skipping...")
            print(prompt, "\n", response.prompt_feedback)
            break
        except exceptions.InternalServerError:
            logger.info(f"Internal server error. Retrying...")
            continue


async def main(
    slices: List[pl.DataFrame],
) -> tuple[List[List[dict | Any]], List[dict]]:
    tasks = []
    input_data = []
    for sliced_df in slices:
        prompt_dict = sliced_df.select(
            pl.col("product_title"),
            pl.col("review_title"),
            pl.col("review_text"),
            pl.col("timestamp"),
        ).to_dicts()
        data = json.dumps(prompt_dict)

        task = asyncio.create_task(async_analyse_reviews(data=data))
        tasks.append(task)
        input_data.append(prompt_dict)

    responses = await asyncio.gather(*tasks)
    return responses, input_data


for iteration in range(3):
    review_files = os.listdir("./review_data/")
    product_files = os.listdir("./product_data/")

    [
        os.remove(f"./review_data/{file}")
        for file in review_files
        if file.endswith("jsonl")
    ]

    [
        os.remove(f"./product_data/{file}")
        for file in product_files
        if file.endswith("jsonl")
    ]

    review_files = sorted(os.listdir("./review_data/"))
    product_files = sorted(os.listdir("./product_data/"))

    for review_file, product_file in zip(review_files, product_files):
        t0 = time.time()

        if iteration in [0, 1] and review_file not in [
            "All_Beauty.jsonl.gz",
            "Amazon_Fashion.jsonl.gz",
            "Appliances.jsonl.gz",
            "Arts_Crafts_and_Sewing.jsonl.gz",
            "Automotive.jsonl.gz",
            "Baby_Products.jsonl.gz",
            "Beauty_and_Personal_Care.jsonl.gz",
            #     "Books.jsonl.gz",
            #     "CDs_and_Vinyl.jsonl.gz",
            #     "Cell_Phones_and_Accessories.jsonl.gz",
            #     "Clothing_Shoes_and_Jewelry.jsonl.gz",
            #     "Digital_Music.jsonl.gz",
            #     "Electronics.jsonl.gz",
            #     "Gift_Cards.jsonl.gz",
            #     "Grocery_and_Gourmet_Food.jsonl.gz",
            #     "Handmade_Products.jsonl.gz",
            #     "Health_and_Household.jsonl.gz",
            #     "Health_and_Personal_Care.jsonl.gz",
            #     #     "Home_and_Kitchen.jsonl.gz",
            #     #     "Industrial_and_Scientific.jsonl.gz",
            #     #     "Kindle_Store.jsonl.gz",
            #     #     "Magazine_Subscriptions.jsonl.gz",
            #     #     "Movies_and_TV.jsonl.gz",
            #     #     "Musical_Instruments.jsonl.gz",
        ]:
            logger.info(
                f"Skipping {review_file}, {product_file} on iteration {iteration}."
            )
            continue

        review_name = review_file.split(".")[0]
        review_path = f"./review_data/{review_name}"

        product_name = product_file.split(".")[0]
        product_path = f"./product_data/{product_name}"

        decompress(file_path_no_ext=review_path)
        decompress(file_path_no_ext=product_path)

        rows = 30
        slice_total = 15
        df = read_data(
            review_path=f"{review_path}.jsonl",
            product_path=f"{product_path}.jsonl",
            slice_init=1_250_000,
            rows=rows,
            slice_total=slice_total,
        )

        slices = []
        for i in range(slice_total):
            if i == 0:
                slices.append(df.slice(offset=i, length=rows))
            else:
                slices.append(df.slice(offset=i * rows, length=rows))

        def get_or_create_eventloop():
            try:
                return asyncio.get_event_loop()
            except RuntimeError as e:
                if "There is no current event loop in thread" in str(e):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return asyncio.get_event_loop()
                else:
                    raise e

        loop = get_or_create_eventloop()
        responses, input_data = loop.run_until_complete(main(slices=slices))

        response_dfs = []
        for response in responses:
            if response is not None:
                try:
                    response_dfs.append(
                        pl.from_dicts(
                            response,
                            schema={"timestamp": pl.UInt64, "evaluation": pl.UInt8},
                        )
                    )
                except TypeError:
                    logger.info("Failed to parse response schema. Skipping...")
                    continue
                except pl.exceptions.ComputeError:
                    logger.info("Failed to parse response schema. Skipping...")
                    continue

        response_df = pl.concat(items=response_dfs)
        final_df = df.join(
            response_df.unique(subset="timestamp"), on="timestamp", how="inner"
        )

        assert len(final_df) > 0, "No data present in final_df."

        final_df.write_delta(target="./export/main/", mode="append")

        logger.info(
            f"""Exported final_df. Relevant row counts were...
        df: {len(df)}
        response_df: {len(response_df)}
        final_df: {len(final_df)}"""
        )

        os.remove(f"{review_path}.jsonl")
        os.remove(f"{product_path}.jsonl")

        logger.info(f"Processing {review_name} took {time.time() - t0}s.")
