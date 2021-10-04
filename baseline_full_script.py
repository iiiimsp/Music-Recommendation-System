#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Usage:
    $ spark-submit --driver-memory=32g --executor-memory=64g --executor-cores=60 basedline_full_script.py <netID>
    $ yarn logs -applicationId <your_application_id> -log_files stdout
'''

import getpass
import pandas as pd
import numpy as np
import itertools


from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import collect_list

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel

from pyspark.mllib.evaluation import RankingMetrics


def main(spark, netID):
    train_path = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet'
    validation_path = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
    test_path = 'hdfs:/user/bm106/pub/MSD/cf_test.parquet'
    
    train = spark.read.parquet(train_path)
    train.createOrReplaceTempView('train')
    
    validation = spark.read.parquet(validation_path)
    validation.createOrReplaceTempView('validation')
    
    test = spark.read.parquet(test_path)
    test.createOrReplaceTempView('test')
    
    
    #Downsample option 1
    df = train.sample(False, 0.25, 42)
    df = train.repartition(50000)
    df.createOrReplaceTempView('df')
        
    #Downsample option 2
    uniq_val = spark.sql('SELECT DISTINCT (user_id) FROM validation')
    uniq_val.createOrReplaceTempView('uniq_val')
    sql_code = """SELECT train.user_id, train.count, train.track_id FROM train 
                  INNER JOIN uniq_val ON train.user_id = uniq_val.user_id"""
    df = spark.sql(sql_code)
    df.createOrReplaceTempView('df')
    
        
    #StringIndexer transform
    indexers = [StringIndexer(inputCol=column, outputCol= 'indexed_' + column).setHandleInvalid('skip') 
                for column in list(set(df.columns)-set(['count','__index_level_0__']))]
    
    pipeline = Pipeline(stages=indexers)
    indexer_transform = pipeline.fit(df)
    
    df = indexer_transform.transform(df)
    validation = indexer_transform.transform(validation)
    test = indexer_transform.transform(test)


        
    #ALS parameters
    rank = [1, 5, 10, 40, 100, 150, 200]  
    reg_params = [0.1, 1]  
    alpha = [1, 10]     
    param_choice = itertools.product(rank, reg_params, alpha)

    #Distinct users from validation
    user_validation = validation.select('indexed_user_id').distinct()

    #True items for evaluation, used for parameter tuning
    group_item = validation.select('indexed_user_id', 'indexed_track_id').groupBy('indexed_user_id')
    true_item = group_item.agg(collect_list('indexed_track_id').alias('true_track'))
                
        
    #Distinct users and true items from test, used after parameter tuning
    user_test = test.select('indexed_user_id').distinct()
    group_item = test.select('indexed_user_id', 'indexed_track_id').groupBy('indexed_user_id')
    true_item = group_item.agg(collect_list('indexed_track_id').alias('true_track'))
    
    
 
    #Hyperparameter tuning
    for i in param_choice:
        als = ALS(rank = i[0], 
                  maxIter = 10, 
                  regParam = i[1], 
                  userCol = 'indexed_user_id',
                  itemCol = 'indexed_track_id',
                  ratingCol = 'count', 
                  implicitPrefs = True, 
                  alpha = i[2],
                  nonnegative=False,
                  coldStartStrategy='drop')
        
        model = als.fit(df)
        print('Result: finished training for {} combination'.format(i))
        
        
        #full model save after parameter tuning
        model.write().overwrite().save(f'hdfs:/user/{netID}/{i[0]}_{i[1]}_{i[2]}_model')
        
        user_factor = model.userFactors
        item_factor = model.itemFactors 
            
        user_factor.toPandas().to_csv(f'hdfs:/user/{netID}/user_factor_{i[0]}_{i[1]}_{i[2]}.csv', index = False,header=True)
        item_factor.toPandas().to_csv(f'hdfs:/user/{netID}/item_factor_{i[0]}_{i[1]}_{i[2]}.csv', index = False,header=True)
        
    
        
        #Evaluate the model by computing the MAP on the validation data
        pred =  model.recommendForUserSubset(user_validation,500)\
                                .select('indexed_user_id','recommendations.indexed_track_id')

        #Convert to rdd for evaluation
        pred_rdd = pred.join(true_item, 'indexed_user_id', 'inner').rdd.map(lambda row: (row[1], row[2]))

        #Get map, precision, ndcg
        metrics = RankingMetrics(pred_rdd)
        map_score = metrics.meanAveragePrecision
        precision = metrics.precisionAt(500)
        ndcg = metrics.ndcgAt(500)
        
        print('map score is: ', map_score)
        print('precision is: ', precision)
        print('ndcg score is: ', ndcg)


    
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()
    netID = getpass.getuser()
    
    # Call our main routine
    main(spark, netID)



