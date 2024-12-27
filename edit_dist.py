import os
from itertools import combinations
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StringType, StructType, StructField

def edit_distance(pair):
    str1, str2 = pair
    # Create a table to store results of subproblems
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Fill dp table
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # Min operations = j
            elif j == 0:
                dp[i][j] = i  # Min operations = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],    # Insert
                                   dp[i - 1][j],    # Remove
                                   dp[i - 1][j - 1])  # Replace

    return dp[m][n]

# Function to compute edit distances using multiple processes
def compute_edit_distance_multiprocess(pair, num_workers):
    # implement a multi-process version of edit_distance function
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(edit_distance, pair_data)
    return results

@udf(returnType=IntegerType())
def edit_distance_udf(str1,str2):
    return edit_distance((str1,str2))

if __name__=="__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--csv_dir', type=str, default='simple-wiki-unique-has-end-punct-sentences.csv', help="Directory of csv file")
    parser.add_argument('--num_sentences', type=int, default=300, help="Number of sentences")
    args = parser.parse_args()

    # Number of processes to use (set to the number of CPU cores)
    num_workers = multiprocessing.cpu_count()
    print(f'number of available cpu cores: {num_workers}')
    # Sample list of string pairs
    text_data = pd.read_csv(args.csv_dir)['sentence']
    text_data = text_data[:args.num_sentences]
    pair_data = list(combinations(text_data, 2))

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Initialize Spark session
    spark = SparkSession.builder.appName("EditDistance").getOrCreate()
    spark_data = spark.createDataFrame([(pair[0], pair[1]) for pair in pair_data], ["str1", "str2"])
    start_time = time.time()
    result = spark_data.select(edit_distance_udf("str1", "str2").alias("distance"))
    spark_distances = result.collect()
    end_time = time.time()
    spark_time = end_time - start_time
    print(f"Time taken (Spark): {spark_time:.3f} seconds")
    spark.stop()
    
    # Stop the Spark session

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Multi-process
    start_time = time.time()
    # Compute edit distances in parallel
    edit_distances = compute_edit_distance_multiprocess(pair_data, num_workers)
    end_time = time.time()
    # print(f"Number of string pairs processed: {len(edit_distances)}")
    print(f"Time taken (multi-process): {end_time - start_time:.3f} seconds")


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Vanilla for loop
    start_time = time.time()
    distances = []
    for pair in tqdm(pair_data, ncols=100):
        distances.append(edit_distance(pair))
    end_time = time.time()
    # print(f"Edit Distances (Non-Spark): {distances}")
    print(f"Time taken (for-loop): {end_time - start_time:.3f} seconds")