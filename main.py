import asyncio
import json
import logging
import math
import os
import time
from functools import reduce

import google.generativeai as genai
import pyspark
import typing_extensions as typing
from delta import *
from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import broadcast, col, concat, monotonically_increasing_id
from pyspark.sql.types import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="main.log", encoding="utf-8", filemode="w", level=logging.INFO
)

builder = (
    SparkSession.builder.appName("one-star-reviews")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "6g")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "2g")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
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


review_files = sorted(os.listdir("./review_data/"))
product_files = sorted(os.listdir("./product_data/"))
review_files = review_files[1:]
product_files = product_files[1:]


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

        try:
            response = await model.generate_content_async(prompt + "\n" + data)
            response = json.loads(response.text)
            return response
        except json.JSONDecodeError:
            logger.info("Failed to parse response. Skipping.")


def split_data(rows: int):
    for i in range(len(review_files)):
        logger.info(f"Processing {review_files[i]} and {product_files[i]}")
        review_df = (
            spark.read.schema(REVIEW_SCHEMA)
            .json(f"./review_data/{review_files[i]}")
            .drop(
                "images",
            )
            .withColumnRenamed("title", "review_title")
            .withColumnRenamed("text", "review_text")
            .repartition("parent_asin")
        )

        product_df = (
            spark.read.schema(PRODUCT_SCHEMA)
            .json(f"./product_data/{product_files[i]}")
            .drop(
                "images",
                "videos",
                # "details",
            )
            .withColumnRenamed("title", "product_title")
            .repartition("parent_asin")
        )

        df = review_df.join(product_df, ["parent_asin"])
        df = (
            df.cache()
            .limit(300000)
            .withColumn("review_id", concat("timestamp", monotonically_increasing_id()))
            .repartition("review_id")
        )

        n_splits = math.ceil(df.count() / rows)
        fractions = [1.0] * n_splits

        dfs = df.randomSplit(fractions, 1)

        review_df.unpersist()
        product_df.unpersist()
        df.unpersist()

        yield dfs


def slice_data(rows: int, slices=3):
    dfs_generator = split_data(rows)
    for dfs in dfs_generator:
        for i in range(len(dfs) // 14):
            lower = slices * i if i == 0 else (slices * i) + 1
            upper = slices * (i + 1)

            logger.info(f"Processing dataframes {lower} - {upper}")
            dfs_slice = dfs[lower:upper]

            yield dfs_slice


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
            ).cache()

            collection = llm_df.collect()
            llm_dict = [row.asDict() for row in collection]
            llm_str = json.dumps(llm_dict)

            task = asyncio.create_task(async_analyse_reviews(llm_str))
            tasks.append(task)

            df.unpersist()
            llm_df.unpersist()

        responses = await asyncio.gather(*tasks)
        return responses

    responses = asyncio.run(main())

    t0 = time.time()
    rdd = spark.sparkContext.parallelize(responses)
    response_df = spark.read.schema(RESPONSE_SCHEMA).json(rdd).cache()

    all_dfs = reduce(DataFrame.union, dfs).cache()

    final_df = all_dfs.join(broadcast(response_df), "review_id", how="inner")
    final_df = final_df.coalesce(1)
    final_df.write.format("delta").mode("append").save("./export/delta-table")

    logger.info(f"Final join and export of data took {time.time() - t0} seconds.")

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
