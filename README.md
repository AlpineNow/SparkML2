# SparkML2

This is a scalable, high-performance machine learning package for `Spark`.
This package is maintained independently from `MLLib` in `Spark`.
There's a chance that some of the algorithms would later get merged with `MLLib`.
The algorithms listed here are going through continuous development, maintenance and updates and thus are provided without guarantees.
Use at your own risk.

This package is provided under the Apache license 2.0.

The current version is `0.2`.

Any comments or questions should be directed to schung@alpinenow.com

The currently available algorithms are:
* [Sequoia Forest](SequoiaForest.md)
* [Gradient Boosted Trees (Under Development)](GradientBoosting.md)

## Compiling the Code

Clone this repository. And run `./sbt assembly` at the project directory. To create an assembly.

## Quick Start (local)
1. Get `Spark` version 1.3.0. A pre-built version can be downloaded from [here](http://www.apache.org/dyn/closer.cgi/spark/spark-1.3.0/spark-1.3.0-bin-cdh4.tgz "SparkDownload")
1a. Untar and set it the location to $SPARK_HOME
2. Clone this repo:
`Git Clone https://github.com/AlpineNow/SparkML2.git`
3. Assemble the jar:
`sbt assembly`
4. Submit Training Job
 ```
rm -rf /tmp/ModelOutputs/mnist && \
$SPARK_HOME/bin/spark-submit \
 --class spark_ml.sequoia_forest.SequoiaForestRunner \
 --name SequoiaForestRunner \
 --driver-memory 4G \
 --executor-memory 4G \
 --num-executors 10 \
target/scala-2.10/spark_ml-assembly-0.1.jar \
 --inputPath data/mnist.tsv.gz \
 --outputPath /tmp/ModelOutputs/mnist \
 --numTrees 100 \
 --numPartitions 10 \
 --labelIndex 780 \
 --checkpointDir /tmp/tree
 ```
5. Submit Prediction Job
```
rm -rf /tmp/ModelOutputs/mnistpredictions && \
$SPARK_HOME/bin/spark-submit \
 --class spark_ml.sequoia_forest.SequoiaForestPredictor \
 --name SequoiaForestPredictor \
 --driver-memory 4G \
 --executor-memory 4G \
 --num-executors 4 \
target/scala-2.10/spark_ml-assembly-0.1.jar \
 --inputPath data/mnist.t.tsv.gz \
 --forestPath /tmp/ModelOutputs/mnist \
 --outputPath /tmp/ModelOutputs/mnistpredictions \
 --labelIndex 780 \
 --outputFieldIndices 780 \
 --pauseDuration 100
```

## Quick Start (for YARN and Linux variants)

1. Get `Spark` version 1.3.0. A pre-built version can be downloaded from [here](https://spark.apache.org/downloads.html "SparkDownload") for some of Hadoop variants. For different Hadoop versions, you'll have to build it after cloning it from github. E.g., to build `Spark` for Apache Hadoop 2.0.5-alpha with `YARN` support, you could do the following.
 1. `git clone https://github.com/apache/spark.git`
 2. `git checkout tags/v1.3.0`
 3. `SPARK_HADOOP_VERSION=2.0.5-alpha SPARK_YARN=true sbt/sbt assembly` or `SPARK_HADOOP_VERSION=2.0.5-alpha SPARK_YARN=true sbt/sbt -Dsbt.override.build.repos=true assembly`
 4. If you want to run this against a different `Spark` version, you should modify `project/build.scala` and change versions of `spark-core` and `spark-mllib` to appropriate versions. Of course, you'll also need to build a matching version of `Spark`.
 5. Additionally, by default, this package builds against `hadoop-client` version `1.0.4`. This will have to change, for instance if you want to build this against different Hadoop versions that are not protocol-compatible with this version. Refer to this `Spark` [page](http://spark.apache.org/docs/latest/building-with-maven.html "SparkMaven") to find out about different Hadoop versions.
2. Clone this repository, and run `./sbt assembly`.
3. In order to connect to Hadoop clusters, you should have Hadoop configurations stored somewhere. E.g., if your Hadoop configurations are stored in `/home/me/hd-config`, then make sure to have the following environment variables.
 * `export HADOOP_CONF_DIR=/home/me/hd-config`
 * `export YARN_CONF_DIR=/home/me/hd-config`
4. Find the location of the `Spark` assembly jar. E.g., it might be `assembly/target/scala-2.10/spark-assembly-1.3.0-hadoop2.0.5-alpha.jar` under the `Spark` directory. Run `export SPARK_JAR=jar_location`.
5. Have some data you want to train on in HDFS. A couple of data sets are provided in this package under the `data` directory for quick testing. E.g., copy `mnist.tsv.gz` and `mnist.t.tsv.gz` to a HDFS directory (E.g. `/Datasets/`).
6. To train a classifier using `YARN`, run the following. `SPARK_DIR` should be replaced with the directory of `Spark` and `SPARK_ML_DIR` should be replaced with the directory where this package resides. 
 * `SPARK_DIR/bin/spark-submit --master yarn --deploy-mode cluster --class spark_ml.sequoia_forest.SequoiaForestRunner --name SequoiaForestRunner --driver-memory 4G --executor-memory 4G --num-executors 10 SPARK_ML_DIR/target/scala-2.10/spark_ml-assembly-0.1.jar --inputPath /Datasets/mnist.tsv.gz --outputPath /ModelOutputs/mnist --numTrees 100 --numPartitions 10 --labelIndex 780`
 * This will train a classification forest with 100 trees using the column 780 as the label and all the other columns as numeric features. The final model would be stored in `/ModelOutputs/mnist` in HDFS.
7. Check status of training through the Hadoop job tracker page. `Spark` also provides its internal progress report when you click on the job's application master link from the job tracker page.
8. Once training is finished, you can predict on a new data set using the following command.
 * `SPARK_DIR/bin/spark-submit --master yarn --deploy-mode cluster --class spark_ml.sequoia_forest.SequoiaForestPredictor --name SequoiaForestPredictor --driver-memory 4G --executor-memory 4G --num-executors 4 SPARK_ML_DIR/target/scala-2.10/spark_ml-assembly-0.1.jar --inputPath /Datasets/mnist.t.tsv.gz --forestPath /ModelOutputs/mnist --outputPath /ModelOutputs/mnistpredictions --labelIndex 780 --outputFieldIndices 780 --pauseDuration 100`
 * The above command would predict on `mnist.t.tsv.gz` using the previously trained model in `/ModelOutputs/mnist` and write predictions under `/ModelOutputs/mnistpredictions`. It'll also write the value of the column 780 (which happens to be the label in this case) along with the predicted value. In the standard output log of the driver, you should also be able to see computed accuracy since the label is given in this case.
9. Training regression requires adding an argument `--forestType Variance`. Likewise, using categorical features requires adding an argument like `--categoricalFeatureIndices 5,6`. This would mean that columns 5 and 6 are to be treated as categorical features. For other options, refer to the command line arguments described below.
