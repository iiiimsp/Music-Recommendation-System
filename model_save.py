#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Save model, itemFactors, userFactors
Usage:
    $ spark-submit --driver-memory=16g --executor-memory=32g --executor-cores=60 model_save.py
    $ yarn logs -applicationId <your_application_id> -log_files stdout
'''

#Use getpass to obtain user netID
import getpass
import pandas as pd
import numpy as np


# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import Row

from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer

import itertools


def model_save(spark,netID):
    train_path = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet'
    
    train = spark.read.parquet(train_path)
    train.createOrReplaceTempView('train')
    df = train.repartition(50000)
    df.createOrReplaceTempView('df')
    
    
    # StringIndexer
    indexers = [StringIndexer(inputCol=column, outputCol= 'indexed_' + column).setHandleInvalid('skip') 
                for column in list(set(df.columns)-set(['count','__index_level_0__']))]
    
    pipeline = Pipeline(stages=indexers)
    indexer_transform = pipeline.fit(df)
    
    df = indexer_transform.transform(df)  
    
    # ALS
    rank = [200] 
    reg_params = [1]  
    alpha = [10]   
    param_choice = itertools.product(rank, reg_params, alpha)

   
    for i in param_choice:
        als = ALS(rank = i[0], 
                  maxIter = 20, 
                  regParam = i[1], 
                  userCol = 'indexed_user_id',
                  itemCol = 'indexed_track_id',
                  ratingCol = 'count', 
                  implicitPrefs = True, 
                  alpha = i[2],
                  nonnegative=False,
                  coldStartStrategy='drop')
        
        model = als.fit(df) 
        model.write().overwrite().save(f'hdfs:/user/{netID}/full_{i[0]}_{i[1]}_{i[2]}_model')
        
        user_factor = model.userFactors
        item_factor = model.itemFactors 
                
        user_factor.toPandas().to_csv(f'hdfs:/user/{netID}/full_user_factor_{i[0]}_{i[1]}_{i[2]}.csv', index = False,header=True)
        item_factor.toPandas().to_csv(f'hdfs:/user/{netID}/full_item_factor_{i[0]}_{i[1]}_{i[2]}.csv', index = False,header=True)
        
        print('Finish saving model for {} combination'.format(i))
        
        
   

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('model_save').getOrCreate()
    netID = getpass.getuser()
    
    # Call our main routine
    model_save(spark,netID)



