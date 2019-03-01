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

class Conf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, output, reducers)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  val reducers =
    opt[Int](descr = "number of reducers", required = false, default = Some(1))
  verify()
}

class PairPartitioner(num: Int) extends Partitioner {
  override def numPartitions: Int = num
  override def getPartition(key: Any): Int = key match {
    case (word1, word2) => (word1.hashCode() & Int.MaxValue) % numPartitions
    case null           => 0
  }
}

object ComputeBigramRelativeFrequencyPairs extends Tokenizer {
  val log = Logger.getLogger(getClass().getName())

  def main(argv: Array[String]) {
    val args = new Conf(argv)

    log.info("Input: " + args.input())
    log.info("Output: " + args.output())
    log.info("Number of reducers: " + args.reducers())

    val conf = new SparkConf().setAppName("ComputeBigramRelativeFrequencyPairs")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    var marginal_total = 0.0

    val textFile = sc.textFile(args.input(), args.reducers())

    val results = textFile
      .flatMap(line => {
        val tokens = tokenize(line)
        if (tokens.length >= 2) {
          val pairs = tokens.sliding(2).map(x => (x.head, x.last)).toList
          val single = tokens.init.map(word => (word, "*")).toList
          List.concat(pairs, single)
        } else {
          List()
        }
      })
      .map(bigram => (bigram, 1))
      .reduceByKey(_ + _, args.reducers())
      .repartitionAndSortWithinPartitions(new PairPartitioner(args.reducers()))
      .map(x => {
        x._1 match {
          case (_, "*") => {
            marginal_total = x._2
            (x._1, marginal_total)
          }
          case (_, _) => {
            (x._1, x._2 / marginal_total)
          }
        }
      })

    results.saveAsTextFile(args.output())
  }
}
