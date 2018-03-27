/* file: SampleKMeans.scala */
//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2017-2018 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================

package DAAL

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

//import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import daal_for_mllib.{KMeans, DAALKMeansModel => KMeansModel}

import org.apache.spark.mllib.linalg.Vectors

object SampleKMeans extends App {
    val conf = new SparkConf().setAppName("Spark KMeans")
    val sc = new SparkContext(conf)

    val data = sc.textFile("/Spark/KMeans/data/KMeans.txt")
    val dataRDD = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    val nClusters = 20
    val nIterations = 10
    val clusters = KMeans.train(dataRDD, nClusters, nIterations, 1, "random")

    val cost = clusters.computeCost(dataRDD)
    println("Sum of squared errors = " + cost)

    sc.stop()
}
