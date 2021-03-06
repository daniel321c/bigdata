/**
  * Bespin: reference implementations of "big data" algorithms
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */
package ca.uwaterloo.cs451.a2;

import io.bespin.scala.util.Tokenizer

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._
import org.apache.spark.Partitioner

import scala.collection.mutable.ListBuffer

class PairsPMIConf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, output, reducers, threshold)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  val reducers =
    opt[Int](descr = "number of reducers", required = false, default = Some(1))
  val threshold =
    opt[Int](descr = "threshold", required = false, default = Some(40))
  verify()
}

object PairsPMI extends Tokenizer {
  val log = Logger.getLogger(getClass().getName())

  def main(argv: Array[String]) {
    val args = new PairsPMIConf(argv)

    log.info("Input: " + args.input())
    log.info("Output: " + args.output())
    log.info("Number of reducers: " + args.reducers())
    log.info("Threshold: " + args.threshold())

    val conf = new SparkConf().setAppName("PairsPMI")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    val textFile = sc.textFile(args.input(), args.reducers())

    val lineCount = textFile.count()

    val threshold = args.threshold()

    val reducers = args.reducers()

    val wordCount = textFile
      .flatMap(line => {
        val tokens = tokenize(line)
        val upperbound = Math.min(40, tokens.length)
        if (upperbound > 0) {
          val uniqueTokens = tokens.take(upperbound).distinct
          uniqueTokens.map(x => (x, 1.0))
        } else {
          List()
        }
      })
      .reduceByKey(_ + _)
      .collectAsMap()

    val bc_wordCount = sc.broadcast(wordCount)

    val pmi = textFile
      .flatMap(line => {
        val listBuffer = ListBuffer[(String, String)]()
        val tokens = tokenize(line)
        val upperbound = Math.min(40, tokens.length)
        if (upperbound > 0) {
          val uniqueTokens = tokens.take(upperbound).distinct
          for (i <- 0 until uniqueTokens.length) {
            for (j <- 0 until uniqueTokens.length) {
              if (i != j) {
                listBuffer += ((uniqueTokens(i), uniqueTokens(j)))
              }
            }
          }
        }
        listBuffer.toList
      })
      .map(p => (p, 1.0))
      .reduceByKey(_ + _, reducers)
      .filter(x => x._2 >= threshold)
      .map(x => {
        var a = bc_wordCount.value(x._1._1)
        var b = bc_wordCount.value(x._1._2)
        var pmi = Math.log10((x._2 * lineCount) / (a * b))
        (x._1, (pmi, x._2.toInt))
      })

    pmi.saveAsTextFile(args.output())
  }
}
