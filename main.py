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
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import broadcast, col, concat, monotonically_increasing_id
from pyspark.sql.types import *
from pyspark.storagelevel import StorageLevel

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="main.log",
    encoding="utf-8",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s: $(levelname)s - %(message)s",
    datefmt="%m/%d/$Y %I:%M%S",
)

review_files = sorted(os.listdir("./review_data/"))
product_files = sorted(os.listdir("./product_data/"))
start = 33
review_files = review_files[start:]
product_files = product_files[start:]

total_size = (os.path.getsize(f"./review_data/{review_files[0]}") / 1000000) + (
    os.path.getsize(f"./product_data/{product_files[0]}") / 1000000
)
shuffle_partitions = int(total_size // 512)

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
        StructField("review_id", StringType(), False),
        StructField("score", IntegerType(), False),
    ]
)


class Reviews(typing.TypedDict):
    review_id: str
    score: int


async def async_analyse_reviews(data: str):
    load_dotenv()
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    with open("./prompt.txt", "r") as file:
        prompt = file.read()

        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[Reviews],
                max_output_tokens=8192,
            ),
        )

    for _ in range(3):
        try:
            response = await model.generate_content_async(prompt + "\n" + data)
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


def read_data():
    logger.info(f"Processing {review_files[0]} and {product_files[0]}")
    review_df = (
        spark.read.schema(REVIEW_SCHEMA)
        .json(f"./review_data/{review_files[0]}")
        .drop(
            "images",
        )
        .withColumnRenamed("title", "review_title")
        .withColumnRenamed("text", "review_text")
        .repartition("parent_asin")
    )

    product_df = (
        spark.read.schema(PRODUCT_SCHEMA)
        .json(f"./product_data/{product_files[0]}")
        .drop(
            "images",
            "videos",
        )
        .withColumnRenamed("title", "product_title")
        .repartition("parent_asin")
    )

    df = review_df.join(product_df, ["parent_asin"])

    cut_df = df.limit(75000)
    df.unpersist()

    cut_df = cut_df.withColumn(
        "review_id", concat("timestamp", monotonically_increasing_id())
    ).repartition("review_id")

    review_df.unpersist()
    product_df.unpersist()

    return cut_df


def slice_data(rows: int, slices=15):
    cut_df = read_data()
    assert cut_df is not None
    cut_df = cut_df.persist(StorageLevel.MEMORY_AND_DISK)

    n_splits = math.ceil(cut_df.count() / rows)
    fractions = [1.0] * n_splits
    dfs = cut_df.randomSplit(fractions, 1)

    for i in range(len(dfs) // 14):
        if i < 25:
            continue
        else:
            lower = slices * i if i == 0 else (slices * i) + 1
            upper = slices * (i + 1)

            logger.info(f"Processing dataframes {lower} - {upper}")
            dfs_slice = dfs[lower:upper]

            if len(dfs_slice) <= 0:
                logger.info("Finished processing all slices.")
                spark.stop()
                exit()

            yield dfs_slice

    cut_df.unpersist()


dfs_generator = slice_data(rows=150)
for dfs in dfs_generator:
    tstart = time.time()

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
                "timestamp",
            )

            collection = llm_df.cache().collect()
            llm_dict = [row.asDict() for row in collection]
            llm_str = json.dumps(llm_dict)

            task = asyncio.create_task(async_analyse_reviews(llm_str))
            tasks.append(task)

            df.unpersist()
            llm_df.unpersist()

        responses = await asyncio.gather(*tasks)
        return responses

    responses = asyncio.run(main())

    rdd = spark.sparkContext.parallelize(responses)
    response_df = spark.read.schema(RESPONSE_SCHEMA).json(rdd).cache()

    all_dfs = reduce(DataFrame.union, dfs).coalesce(6).cache()

    final_df = all_dfs.join(broadcast(response_df.distinct()), "review_id", how="inner")
    final_df = final_df.coalesce(1)
    assert (
        final_df.filter(col("review_id").isNotNull()).count() > 0
    ), f"No data present in final_df: {final_df.filter(col('review_id').isNotNull()).count()}"
    final_df.write.format("delta").mode("append").save("./export/delta-table")

    logger.info(
        f"""Exported final_df. Relevant row counts were...
    all_dfs: {all_dfs.filter(col("review_id").isNotNull()).count()}
    response_df: {response_df.filter(col("review_id").isNotNull()).count()}
    final_df: {final_df.filter(col("review_id").isNotNull()).count()}"""
    )
    logger.info(
        f"Overall processing time for slice was {time.time() - tstart} seconds."
    )

    rdd.unpersist()
    [df.unpersist() for df in dfs]
    response_df.unpersist()
    all_dfs.unpersist()
    final_df.unpersist()
