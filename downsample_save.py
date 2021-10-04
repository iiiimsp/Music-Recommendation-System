#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Backup train files with repartition
Usage:
    $ spark-submit downsample_save.py <student_netID>
'''

#Use getpass to obtain user netID

import getpass
import sys
import pandas as pd
import numpy as np


# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession



def downsample_save(spark, netID):
    
    train_path = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet'
    validation_path = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
    
    train = spark.read.parquet(train_path)
    train.createOrReplaceTempView('train')
    
    validation = spark.read.parquet(validation_path)
    validation.createOrReplaceTempView('validation')
    

    
    #option 1: save 25% training sample
    df = train.sample(False, 0.25 , 42)
    df.createOrReplaceTempView('df')
    df = df.repartition(10000)
    df.write.mode('overwrite').parquet(f'hdfs:/user/{netID}/train_25.parquet')
    
    #option 2
    uniq_val = spark.sql('SELECT DISTINCT (user_id) FROM validation')
    uniq_val.createOrReplaceTempView('uniq_val')
    sql_code = """SELECT train.user_id, train.count, train.track_id FROM train 
                  INNER JOIN uniq_val ON train.user_id = uniq_val.user_id"""
    df = spark.sql(sql_code)
    df.createOrReplaceTempView('df')
    df.write.mode('overwrite').parquet(f'hdfs:/user/{netID}/sql_downsample.parquet')
      

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('downsample').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()
    
    
    downsample_save(spark, netID)

    # Call our main routine
