import java_bootstrap
java_bootstrap.ensure_java()

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CustomerSegmentation").getOrCreate()