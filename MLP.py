import os
import argparse
import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims):
        super(MLPClassifier, self).__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.model(x)
        predicted_classes = torch.argmax(logits, dim=1)
        return predicted_classes

# Define a pandas UDF for parallel classification
@pandas_udf(IntegerType())
def MLPClassifier_udf(*batch_inputs):
    batch_inputs = torch.tensor(np.column_stack(batch_inputs), dtype=torch.float32)
    predictions = mlp_model(batch_inputs).cpu().numpy()
    return pd.Series(predictions.flatten())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP Inference with PySpark")
    parser.add_argument('--n_input', type=int, default=10000, help="Number of input samples")
    parser.add_argument('--hidden_dim', type=int, default=1024, help="Hidden dimension size")
    parser.add_argument('--hidden_layer', type=int, default=50, help="Number of hidden layers")
    args = parser.parse_args()

    input_dim = 128
    num_classes = 10
    hidden_dims = [args.hidden_dim] * args.hidden_layer

    # Model and input setup
    mlp_model = MLPClassifier(input_dim, num_classes, hidden_dims)
    mlp_model = mlp_model.cpu()  # Ensure model is on CPU
    x = torch.randn(args.n_input, input_dim)

    # Initialize Spark session with adjusted configuration
    spark = SparkSession.builder.appName("MLPInference") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.maxResultSize", "5G") \
        .getOrCreate()

    # Convert PyTorch tensor to Pandas DataFrame and then to Spark DataFrame
    df = pd.DataFrame(x.numpy())
    sdf = spark.createDataFrame(df).repartition(1000)  # Increased partitions for better parallelism

    # Apply the UDF to perform distributed classification
    start_time = time.time()
    result_sdf = sdf.select(MLPClassifier_udf(*sdf.columns).alias("predictions"))
    end_time = time.time()

    spark_time = end_time - start_time
    print(f"Time taken for distributed classification: {spark_time:.6f} seconds")

    spark.stop()

    # Non-spark version for comparison
    start_time = time.time()
    output = mlp_model(x)
    end_time = time.time()

    non_spark_time = end_time - start_time

    print(f"Output shape: {output.shape}")
    print(f"Time taken for forward pass: {non_spark_time:.6f} seconds")

    print(f"Time cost for spark and non-spark version: [{spark_time:.6f}, {non_spark_time:.6f}] seconds")