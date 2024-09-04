import pyspark
from pyspark.sql import SparkSession, dataframe
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


def extract_data(review_path: str, product_path: str, size: int) -> dataframe.DataFrame:
    review_df = spark.read.schema(REVIEW_SCHEMA).json(review_path)
    review_df = review_df.drop("images")
    review_df = review_df.withColumnRenamed("title", "review_title").withColumnRenamed(
        "text", "review_text"
    )

    product_df = spark.read.schema(PRODUCT_SCHEMA).json(product_path)
    product_df = product_df.drop("images", "videos")
    product_df = product_df.withColumnRenamed("title", "product_title")

    df = review_df.join(product_df, ["parent_asin"])
    df = df.limit(size)

    return df


beauty_df = extract_data(
    review_path="./review_data/All_Beauty.jsonl.gz",
    product_path="./product_data/meta_All_Beauty.jsonl.gz",
    size=10,
)

beauty_df.show(vertical=True)
