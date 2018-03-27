/* file: SamplePCA.scala */
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

//import org.apache.spark.mllib.feature.{PCA, PCAModel}
import daal_for_mllib.{PCA, PCAModel}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import java.io._

object SamplePCA extends App {
    val conf = new SparkConf().setAppName("Spark PCA")
    val sc = new SparkContext(conf)

    val data = sc.textFile("/Spark/PCA/data/PCA.txt")
    val dataRDD = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    val model = new PCA(10).fit(dataRDD)

    println("Eigen vectors:")
    println(model.pc.toString())
    println("")
    println("Eigen values:")
    println(model.explainedVariance.toString())

    sc.stop()
}
