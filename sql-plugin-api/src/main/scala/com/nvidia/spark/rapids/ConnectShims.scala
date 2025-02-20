package com.nvidia.spark.rapids

trait ConnectShims {
    type Strategy = org.apache.spark.sql.execution.SparkStrategy 
}
