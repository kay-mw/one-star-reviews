import json
import os

import pyspark
from dotenv import load_dotenv
from groq import Groq
from pyspark.sql import SparkSession, functions
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

spark = SparkSession.builder.appName("one-star-reviews").getOrCreate()

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


review_files = sorted(os.listdir("./review_data/"))
product_files = sorted(os.listdir("./product_data/"))


def extract_data(size: int):
    for i in range(len(review_files)):
        review_df = spark.read.schema(REVIEW_SCHEMA).json(
            f"review_data/{review_files[i]}"
        )
        review_df = review_df.drop("images")
        review_df = review_df.withColumnRenamed(
            "title", "review_title"
        ).withColumnRenamed("text", "review_text")
        review_df = review_df.withColumn(
            "review_id", functions.monotonically_increasing_id()
        )

        product_df = spark.read.schema(PRODUCT_SCHEMA).json(
            f"product_data/{product_files[i]}"
        )
        product_df = product_df.drop("images", "videos")
        product_df = product_df.withColumnRenamed("title", "product_title")

        df = review_df.join(product_df, ["parent_asin"])
        df = df.limit(size)

        yield df


def analyse_reviews(context: str):
    load_dotenv()
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    with open("./prompt.txt", "r") as file:
        prompt = " ".join(line.rstrip() for line in file)
        prompt = prompt + context

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert review analyzer who evalutes the quality and usefulness of product reviews in JSON.",
                },
                {"role": "user", "content": f"{prompt}"},
            ],
            model="llama-3.1-70b-versatile",
            response_format={"type": "json_object"},
        )

    data = chat_completion.choices[0].message.content

    assert data is not None

    data = json.loads(data)

    return data


inner_schema = StructType(
    [
        StructField("informativeness", IntegerType(), True),
        StructField("objectivity", IntegerType(), True),
        StructField("key_points", StringType(), True),
        StructField("relevance", StringType(), True),
        StructField("score", IntegerType(), True),
    ]
)

schema = MapType(StringType(), inner_schema)

df_generator = extract_data(size=10)
for df in df_generator:
    df.cache()
    json_df = df.toJSON().collect()
    json_df = [json.loads(x) for x in json_df]
    json_df = json.dumps(json_df)
    response = analyse_reviews(context=json_df)
    df.unpersist()
    response_df = spark.createDataFrame([response], schema=schema)
    response_df = response_df.selectExpr("explode(value) as (id, details)")
    response_df = response_df.select(
        "id",
        "details.informativeness",
        "details.objectivity",
        "details.key_points",
        "details.relevance",
        "details.score",
    )
    response_df.cache()
    response_df.show()
    response_df.unpersist()
