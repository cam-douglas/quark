# Apache Spark - Distributed Data Processing

## Overview
Apache Spark is the most powerful open-source distributed computing engine for large-scale data processing and analytics.

## Key Features
- **In-Memory Computing**: 100x faster than Hadoop MapReduce
- **Unified Platform**: SQL, Streaming, ML, Graph processing
- **Fault Tolerance**: Automatic recovery from failures
- **Scalability**: Handle petabytes of data across clusters
- **Language Support**: Python, Java, Scala, R

## Production Setup

### 1. Spark Cluster Configuration
```yaml
# spark-cluster-config.yaml
spark:
  version: "3.5.0"
  master:
    instance_type: "m5.2xlarge"
    replicas: 1
  workers:
    instance_type: "m5.4xlarge"
    min_replicas: 2
    max_replicas: 20
    autoscaling: true
  
  storage:
    type: "s3"  # or hdfs, gcs, azure
    bucket: "spark-data-bucket"
    
  monitoring:
    prometheus: true
    grafana: true
    jupyter: true
```

### 2. Kubernetes Deployment
```yaml
# spark-k8s-deployment.yaml
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: spark-ml-job
  namespace: spark
spec:
  type: Python
  pythonVersion: "3"
  mode: cluster
  image: "spark:3.5.0"
  imagePullPolicy: Always
  
  sparkVersion: "3.5.0"
  
  restartPolicy:
    type: OnFailure
    onFailureRetries: 3
    onFailureRetryInterval: 10
    onSubmissionFailureRetries: 5
    onSubmissionFailureRetryInterval: 20
  
  driver:
    cores: 1
    coreLimit: "1200m"
    memory: "512m"
    labels:
      version: 3.5.0
    serviceAccount: spark
  
  executor:
    cores: 2
    instances: 2
    memory: "512m"
    labels:
      version: 3.5.0
  
  mainApplicationFile: "s3a://bucket/spark-jobs/ml_pipeline.py"
  arguments:
    - "s3a://bucket/input/data"
    - "s3a://bucket/output/results"
```

## ML/AI Workloads

### 1. Distributed ML Training
```python
# distributed_ml_training.py
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Distributed ML Training") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Load data
df = spark.read.parquet("s3a://bucket/input/data")

# Feature engineering
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3"],
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=True
)

# Model training
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="scaled_features",
    numTrees=100,
    maxDepth=10
)

# Pipeline
pipeline = Pipeline(stages=[assembler, scaler, rf])

# Train model
model = pipeline.fit(df)

# Save model
model.write().overwrite().save("s3a://bucket/models/random_forest")

# Evaluate model
predictions = model.transform(df)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Model accuracy: {accuracy}")

spark.stop()
```

### 2. Real-time Streaming
```python
# real_time_streaming.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Real-time Streaming") \
    .config("spark.sql.streaming.checkpointLocation", "s3a://bucket/checkpoints") \
    .getOrCreate()

# Define schema
schema = StructType([
    StructField("timestamp", TimestampType(), True),
    StructField("user_id", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("value", DoubleType(), True)
])

# Read streaming data
streaming_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "events") \
    .load()

# Parse JSON data
parsed_df = streaming_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Process streaming data
processed_df = parsed_df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window("timestamp", "5 minutes"),
        "event_type"
    ) \
    .agg(
        count("*").alias("event_count"),
        avg("value").alias("avg_value")
    )

# Write to output
query = processed_df.writeStream \
    .format("parquet") \
    .option("path", "s3a://bucket/output/streaming") \
    .option("checkpointLocation", "s3a://bucket/checkpoints/streaming") \
    .outputMode("append") \
    .trigger(processingTime="1 minute") \
    .start()

query.awaitTermination()
```

### 3. Large-scale Data Processing
```python
# large_scale_processing.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Large-scale Data Processing") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
    .getOrCreate()

# Load large datasets
users_df = spark.read.parquet("s3a://bucket/users")
transactions_df = spark.read.parquet("s3a://bucket/transactions")
products_df = spark.read.parquet("s3a://bucket/products")

# Join datasets
joined_df = transactions_df \
    .join(users_df, "user_id") \
    .join(products_df, "product_id")

# Complex aggregations
window_spec = Window.partitionBy("user_id").orderBy("timestamp")

user_metrics = joined_df \
    .withColumn("transaction_rank", rank().over(window_spec)) \
    .withColumn("cumulative_amount", sum("amount").over(window_spec)) \
    .groupBy("user_id") \
    .agg(
        count("*").alias("total_transactions"),
        sum("amount").alias("total_spent"),
        avg("amount").alias("avg_transaction"),
        max("amount").alias("max_transaction"),
        collect_list("product_category").alias("categories")
    )

# Write results
user_metrics.write.mode("overwrite").parquet("s3a://bucket/output/user_metrics")

spark.stop()
```

## Performance Optimization

### 1. Memory Management
```python
# memory_optimization.py
spark = SparkSession.builder \
    .appName("Memory Optimization") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m") \
    .config("spark.sql.adaptive.minNumPostShufflePartitions", "1") \
    .config("spark.sql.adaptive.maxNumPostShufflePartitions", "200") \
    .config("spark.sql.adaptive.shuffle.targetPostShuffleInputSize", "64m") \
    .config("spark.sql.adaptive.shuffle.minNumPostShufflePartitions", "1") \
    .config("spark.sql.adaptive.shuffle.maxNumPostShufflePartitions", "200") \
    .getOrCreate()
```

### 2. Partitioning Strategy
```python
# partitioning_strategy.py
# Repartition data for optimal processing
df = df.repartition(100)  # Based on cluster size

# Partition by key for joins
df = df.repartitionByRange(100, "user_id")

# Coalesce to reduce partitions
df = df.coalesce(50)
```

### 3. Caching Strategy
```python
# caching_strategy.py
# Cache frequently used DataFrames
df.cache()
df.persist()

# Unpersist when no longer needed
df.unpersist()
```

## Monitoring & Observability

### 1. Spark UI Configuration
```yaml
# spark-ui-config.yaml
spark:
  ui:
    port: 4040
    bindAddress: "0.0.0.0"
    retention: "24h"
    
  eventLog:
    enabled: true
    dir: "s3a://bucket/spark-events"
    
  metrics:
    enabled: true
    prometheus: true
```

### 2. Custom Metrics
```python
# custom_metrics.py
from pyspark.sql import SparkSession
import time

spark = SparkSession.builder \
    .appName("Custom Metrics") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Custom accumulator for tracking
from pyspark import AccumulatorParam

class MetricsAccumulator(AccumulatorParam):
    def zero(self, initialValue):
        return {"count": 0, "sum": 0.0, "start_time": time.time()}
    
    def addInPlace(self, v1, v2):
        v1["count"] += v2["count"]
        v1["sum"] += v2["sum"]
        return v1

metrics = spark.sparkContext.accumulator(
    {"count": 0, "sum": 0.0, "start_time": time.time()},
    MetricsAccumulator()
)

# Use in transformations
def process_row(row):
    metrics.add({"count": 1, "sum": row.value, "start_time": 0})
    return row

df.rdd.map(process_row).collect()
print(f"Processed {metrics.value['count']} rows")
```

## Integration with ML Tools

### 1. MLflow Integration
```python
# mlflow_integration.py
import mlflow
import mlflow.spark
from pyspark.ml import Pipeline

# Start MLflow run
with mlflow.start_run():
    # Train model
    pipeline = Pipeline(stages=[assembler, scaler, rf])
    model = pipeline.fit(df)
    
    # Log model
    mlflow.spark.log_model(model, "spark-model")
    
    # Log metrics
    predictions = model.transform(df)
    accuracy = evaluator.evaluate(predictions)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log parameters
    mlflow.log_param("num_trees", 100)
    mlflow.log_param("max_depth", 10)
```

### 2. Delta Lake Integration
```python
# delta_lake_integration.py
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Delta Lake Integration") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Write to Delta format
df.write.format("delta").mode("overwrite").save("s3a://bucket/delta/table")

# Read from Delta format
delta_df = spark.read.format("delta").load("s3a://bucket/delta/table")

# Time travel
historical_df = spark.read.format("delta").option("versionAsOf", 0).load("s3a://bucket/delta/table")
```

## Best Practices

### 1. Performance
- **Partitioning**: Use appropriate partition sizes (64MB-128MB)
- **Caching**: Cache frequently accessed DataFrames
- **Broadcasting**: Use broadcast joins for small tables
- **Skew Handling**: Handle data skew with salting techniques

### 2. Resource Management
- **Memory Tuning**: Configure executor memory and cores
- **Dynamic Allocation**: Enable dynamic resource allocation
- **Speculative Execution**: Enable for straggler tasks
- **Backpressure**: Configure streaming backpressure

### 3. Monitoring
- **Spark UI**: Monitor job progress and resource usage
- **Metrics**: Collect custom metrics for business logic
- **Logging**: Centralize logs for debugging
- **Alerting**: Set up alerts for job failures

### 4. Cost Optimization
- **Spot Instances**: Use for batch processing
- **Auto-scaling**: Scale clusters based on workload
- **Data Locality**: Keep data close to compute
- **Compression**: Use efficient compression formats
