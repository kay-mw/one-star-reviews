# TODO: Re-download that review file I deleted.
# TODO: Extend training dataset for improved fine-tuning.

import gzip
import json
import logging
import os
import shutil
import time
from typing import List

import polars as pl
from transformers import TextStreamer
from unsloth import FastLanguageModel

os.environ["POLARS_MAX_THREADS"] = "6"

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


def decompress(file_path_no_ext: str, buffer_size: int) -> None:
    with gzip.open(f"{file_path_no_ext}.jsonl.gz", "rb") as f_in:
        with open(f"{file_path_no_ext}.jsonl", "wb") as f_out:
            shutil.copyfileobj(fsrc=f_in, fdst=f_out, length=buffer_size)


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


def prep_model(max_seq_length: int, dtype=None, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./tuning/review-model/",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def prompt_model(
    model,
    tokenizer,
    slices: List[pl.DataFrame],
):
    responses = []
    for sliced_df in slices:
        prompt_dict = sliced_df.select(
            pl.col("product_title"),
            pl.col("rating"),
            pl.col("review_text"),
            pl.col("review_title"),
            pl.col("timestamp"),
        ).to_dicts()

        with open("prompt.md", "r") as file:
            instruction = file.read()

        prompt = instruction + json.dumps(prompt_dict)
        messages = [{"role": "user", "content": prompt}]
        print(messages)

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        tensor_response = model.generate(
            input_ids,
            streamer=text_streamer,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=300,
        )

        response = tokenizer.batch_decode(tensor_response)[0]
        _, match, after = response.partition("### Response:")

        if match:
            result = match + after
        else:
            raise ValueError("Failed to parse response.")

        result_lines = result.splitlines()[2:-2]
        try:
            result_json = json.loads("".join(result_lines))
            responses.append(result_json)
        except json.JSONDecodeError:
            logger.info("Failed to parse response, skipping...")
            responses.append(None)
            continue

    return responses


[os.remove(f"./review_data/{file}") for file in review_files if file.endswith("jsonl")]

[
    os.remove(f"./product_data/{file}")
    for file in product_files
    if file.endswith("jsonl")
]

open("prompt_examples.json", "w").close()

# model, tokenizer = prep_model(max_seq_length=512)
for review_file, product_file in zip(review_files, product_files):
    t0 = time.time()

    review_name = review_file.split(".")[0]
    review_path = f"./review_data/{review_name}"

    product_name = product_file.split(".")[0]
    product_path = f"./product_data/{product_name}"

    decompress(file_path_no_ext=review_path, buffer_size=1 * 1000000)
    decompress(file_path_no_ext=product_path, buffer_size=1 * 1000000)

    rows = 10
    slice_total = 1
    seed = 20
    df = read_data(
        review_path=f"{review_path}.jsonl",
        product_path=f"{product_path}.jsonl",
        slice_init=1000000,
        rows=rows,
        slice_total=slice_total,
        seed=seed,
    )

    prompt_dict = df.select(
        pl.col("product_title"),
        pl.col("rating"),
        pl.col("review_text"),
        pl.col("review_title"),
        pl.col("timestamp"),
    ).to_dicts()
    with open("prompt_examples.json", "a") as file:
        file.write(json.dumps(prompt_dict) + "\n")

    # slices = []
    # for i in range(slice_total):
    #     slices.append(df.slice(offset=i * rows, length=rows))
    #
    # responses = prompt_model(model=model, tokenizer=tokenizer, slices=slices)
    # print("DONE!")
    # time.sleep(60)
    #
    # response_dfs = []
    # for response in responses:
    #     if response is not None:
    #         response_dfs.append(
    #             pl.from_dicts(
    #                 response, schema={"timestamp": pl.UInt64, "score": pl.UInt8}
    #             )
    #         )
    #
    # response_df = pl.concat(items=response_dfs)
    # final_df = df.join(response_df, on="timestamp", how="inner")
    #
    # with pl.Config(set_fmt_str_lengths=500, tbl_rows=-1):
    #     print(
    #         final_df.select(
    #             pl.col("rating"),
    #             pl.col("review_title"),
    #             pl.col("review_text"),
    #             pl.col("score"),
    #         )
    #     )
    #
    os.remove(f"{review_path}.jsonl")
    os.remove(f"{product_path}.jsonl")

    # assert len(final_df) > 0, "No data present in final_df."
    #
    # # final_df.write_delta(target="./export/polars-delta/", mode="append")
    #
    # logger.info(
    #     f"""Exported final_df. Relevant row counts were...
    # df: {len(df)}
    # response_df: {len(response_df)}
    # final_df: {len(final_df)}"""
    # )
    #

    logger.info(f"Processing {review_name} took {time.time() - t0}s.")
