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

class StripesConf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, output, reducers)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  val reducers =
    opt[Int](descr = "number of reducers", required = false, default = Some(1))
  verify()
}

object ComputeBigramRelativeFrequencyStripes extends Tokenizer {
  val log = Logger.getLogger(getClass().getName())

  def main(argv: Array[String]) {
    val args = new StripesConf(argv)

    log.info("Input: " + args.input())
    log.info("Output: " + args.output())
    log.info("Number of reducers: " + args.reducers())

    val conf =
      new SparkConf().setAppName("ComputeBigramRelativeFrequencyStripes")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    var marginal_total = 0.0

    val textFile = sc.textFile(args.input(), args.reducers())

    val reducers = args.reducers()

    val results = textFile
      .flatMap(line => {
        val tokens = tokenize(line)
        if (tokens.length >= 2) {
          tokens.sliding(2).map(x => { (x.head, Map(x.last -> 1.0)) })
        } else {
          List()
        }
      })
      .reduceByKey((a, b) => {
        a ++ b.map { case (k, v) => k -> (v + a.getOrElse(k, 0.0)) }
      }, reducers)
      .map(x => {
        var sum = x._2.values.sum
        val ratioMap = x._2.map { case (k, v) => k -> (v / sum) }
        (x._1, ratioMap)
      })

    results.saveAsTextFile(args.output())
  }
}
