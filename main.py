import asyncio
import json
import logging
import math
import os
import time
from functools import reduce

import google.generativeai as genai
import typing_extensions as typing
from delta import *
from dotenv import load_dotenv
from google.api_core import exceptions
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import broadcast, col
from pyspark.sql.types import *

# NOTE: If I want this to work, I _could_ just start and stop Spark on each iteration...

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

total_size = (os.path.getsize(f"./review_data/{review_files[0]}") / 1000000) + (
    os.path.getsize(f"./product_data/{product_files[0]}") / 1000000
)
shuffle_partitions = int(total_size // 1000)
if shuffle_partitions < 2:
    shuffle_partitions = 4

builder = (
    SparkSession.builder.appName("one-star-reviews")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    .config("spark.driver.memory", "6g")
    .config("spark.driver.cores", "2")
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.executor.memory", "4g")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "2g")
    .config("spark.sql.shuffle.partitions", f"{shuffle_partitions}")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()


PRODUCT_SCHEMA = StructType(
    [
        StructField("main_category", StringType(), True),
        StructField("title", StringType(), True),
        StructField("average_rating", FloatType(), True),
        StructField("rating_number", IntegerType(), True),
        StructField("features", ArrayType(StringType()), True),
        StructField("description", ArrayType(StringType()), True),
        StructField("price", FloatType(), True),
        StructField("images", ArrayType(StringType()), True),
        StructField("videos", ArrayType(StringType()), True),
        StructField("store", StringType(), True),
        StructField("categories", ArrayType(StringType()), True),
        StructField("details", MapType(StringType(), StringType()), True),
        StructField("parent_asin", StringType(), True),
        StructField("bought_together", ArrayType(StringType()), True),
    ]
)
REVIEW_SCHEMA = StructType(
    [
        StructField("rating", FloatType(), True),
        StructField("title", StringType(), False),
        StructField("text", StringType(), False),
        StructField("images", ArrayType(StringType()), True),
        StructField("asin", StringType(), True),
        StructField("parent_asin", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("timestamp", LongType(), True),
        StructField("verified_purchase", BooleanType(), True),
        StructField("helpful_vote", IntegerType(), True),
    ]
)
RESPONSE_SCHEMA = StructType(
    [
        StructField("timestamp", LongType(), False),
        StructField("score", IntegerType(), False),
    ]
)


def read_data(review_path: str, product_path: str, rows: int, slice_total: int, seed=1):
    logger.info(f"Processing {review_path}, {product_path}...")
    logger.info(f"Sample seed is {seed}.")

    review_df = (
        spark.read.schema(REVIEW_SCHEMA)
        .json(f"./review_data/{review_path}")
        .drop(
            "images",
        )
        .withColumnRenamed("title", "review_title")
        .withColumnRenamed("text", "review_text")
        .repartition("parent_asin")
    )

    product_df = (
        spark.read.schema(PRODUCT_SCHEMA)
        .json(f"./product_data/{product_path}")
        .drop(
            "images",
            "videos",
        )
        .withColumnRenamed("title", "product_title")
        .repartition("parent_asin")
    )

    df = review_df.join(product_df, "parent_asin")
    review_df.unpersist(True)
    product_df.unpersist(True)

    row_target = rows * slice_total
    frac = row_target / df.count()

    sample = df.sample(fraction=frac, seed=seed)
    logger.info(f"Sample row count is {sample.count()}.")

    df.unpersist(True)
    sample.unpersist(True)

    return sample


def slice_data(sample, slice_size: int, size: int):
    assert slice_total % slice_size == 0, "slice_total is not divisible by slice_size"

    n_splits = math.ceil(size / rows)
    fractions = [1.0] * n_splits
    samples = sample.randomSplit(fractions, 1)
    logger.info(f"dfs contains {len(samples)} slices.")

    for i in range(len(samples) // slice_size):
        lower = slice_size * i if i == 0 else (slice_size * i) + 1
        upper = slice_size * (i + 1)

        logger.info(f"Processing dataframes {lower} - {upper}")
        dfs_slice = samples[lower:upper]

        yield dfs_slice


class Reviews(typing.TypedDict):
    timestamp: int
    score: int


async def async_analyse_reviews(data: str):
    load_dotenv()
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    with open("./prompt.md", "r") as file:
        prompt = file.read() + "\n" + data
        p_file = open("prompt_examples.json", "a")
        p_file.write(data + "\n\n")
        p_file.close()

        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[Reviews],
            ),
        )

    for _ in range(3):
        try:
            response = await model.generate_content_async(prompt)
            response = json.loads(response.text)
            return response
        except json.JSONDecodeError:
            logger.info("Failed to parse response. Skipping...")
            break
        except exceptions.ResourceExhausted:
            n_seconds = 3
            logger.info(f"Rate limited. Waiting {n_seconds}s and retrying...")
            time.sleep(n_seconds)
            continue


rows = 10
slice_size = 10
slice_total = 10
for review_path, product_path in zip(review_files, product_files):
    tstart = time.time()
    sample = read_data(
        review_path=review_path,
        product_path=product_path,
        rows=rows,
        slice_total=slice_total,
    ).persist(StorageLevel.MEMORY_AND_DISK)
    size = sample.count()
    dfs_slice = slice_data(sample=sample, slice_size=slice_size, size=size)
    sample.unpersist(True)

    for dfs in dfs_slice:

        async def main():
            tasks = []
            for df in dfs:
                llm_df = df.drop(
                    "parent_asin",
                    "asin",
                    "user_id",
                    "verified_purchase",
                    "helpful_vote",
                    "bought_together",
                    "details",
                    "categories",
                    "store",
                    "description",
                    "features",
                    "main_category",
                    "average_rating",
                    "rating_number",
                    "price",
                )

                collection = llm_df.persist(StorageLevel.MEMORY_AND_DISK).collect()
                llm_dict = [row.asDict() for row in collection]
                llm_str = json.dumps(llm_dict)

                task = asyncio.create_task(async_analyse_reviews(llm_str))
                tasks.append(task)

                df.unpersist(True)
                llm_df.unpersist(True)

            responses = await asyncio.gather(*tasks)
            return responses

        responses = asyncio.run(main())
        logger.info(
            f"Overall processing time for slice was {time.time() - tstart} seconds."
        )

        rdd = spark.sparkContext.parallelize(responses)
        response_df = (
            spark.read.schema(RESPONSE_SCHEMA)
            .json(rdd)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        all_dfs = (
            reduce(DataFrame.union, dfs)
            .coalesce(2)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        final_df = all_dfs.join(
            broadcast(response_df.distinct()), "timestamp", how="inner"
        )

        assert (
            final_df.filter(col("timestamp").isNotNull()).count() > 0
        ), f"No data present in final_df: {final_df.filter(col('timestamp').isNotNull()).count()}"

        final_df = final_df.coalesce(1)
        final_df.write.format("delta").mode("append").save("./export/few")
        final_df.unpersist(True)

        logger.info(
            f"""Exported final_df. Relevant row counts were...
        all_dfs: {all_dfs.filter(col("timestamp").isNotNull()).count()}
        response_df: {response_df.filter(col("timestamp").isNotNull()).count()}
        final_df: {final_df.filter(col("timestamp").isNotNull()).count()}"""
        )

        rdd.unpersist(True)
        [df.unpersist(True) for df in dfs]
        response_df.unpersist(True)
        all_dfs.unpersist(True)

spark.stop()
