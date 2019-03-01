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

package ca.uwaterloo.cs451.a3;

import io.bespin.java.util.Tokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.commons.net.tftp.TFTPPacket;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MapFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import tl.lin.data.pair.PairOfStringInt;
import tl.lin.data.map.HMapStIW;
import tl.lin.data.map.MapKI;

import java.io.IOException;
import java.util.*;

import java.io.*;

public class BuildInvertedIndexCompressed extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BuildInvertedIndexCompressed.class);

    private static final class MyMapper extends Mapper<LongWritable, Text, PairOfStringInt, IntWritable> {

        private static final HMapStIW MAP = new HMapStIW();
        private static final IntWritable tf = new IntWritable();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            List<String> tokens = Tokenizer.tokenize(value.toString());

            MAP.clear();

            for (String token : tokens) {
                MAP.increment(token);
            }

            for (MapKI.Entry<String> keyValue : MAP.entrySet()) {
                tf.set(keyValue.getValue());
                context.write(new PairOfStringInt(keyValue.getKey(), (int) key.get()), tf);
            }
        }
    }

    private static final class MyReducer extends Reducer<PairOfStringInt, IntWritable, Text, BytesWritable> {

        private static Text WORD = new Text();
        private final static ByteArrayOutputStream bos = new ByteArrayOutputStream();
        private final static DataOutputStream dos = new DataOutputStream(bos);

        ByteArrayOutputStream tmp_bos = new ByteArrayOutputStream();
        DataOutputStream tmp_dos = new DataOutputStream(tmp_bos);

        String prevWord = "";
        int df = 0;
        int prevLine = 0;

        @Override
        public void reduce(PairOfStringInt key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            if (!key.getLeftElement().equals(prevWord) && !prevWord.equals("")) {
                // write array
                dos.flush();
                bos.flush();

                // write df first
                WritableUtils.writeVInt(tmp_dos, df);
                tmp_dos.write(bos.toByteArray());

                tmp_dos.flush();

                WORD.set(prevWord);
                context.write(WORD, new BytesWritable(tmp_bos.toByteArray()));

                // reset
                df = 0;
                prevLine = 0;
                dos.flush();
                bos.reset();
                tmp_dos.flush();
                tmp_bos.reset();
            }

            Iterator<IntWritable> iter = values.iterator();

            df++;
            prevWord = key.getLeftElement();
            int currentLine = key.getRightElement();
            int lineGap = currentLine - prevLine;
            prevLine = currentLine;
            WritableUtils.writeVInt(dos, lineGap);
            WritableUtils.writeVInt(dos, iter.next().get());

        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            dos.flush();
            bos.flush();

            WritableUtils.writeVInt(tmp_dos, df);
            tmp_dos.write(bos.toByteArray());

            tmp_dos.flush();

            WORD.set(prevWord);
            context.write(WORD, new BytesWritable(tmp_bos.toByteArray()));

            // reset
            df = 0;
            prevLine = 0;
            bos.close();
            dos.close();
            tmp_bos.close();
            tmp_bos.close();
        }

    }

    protected static class MyPartitioner extends Partitioner<PairOfStringInt, IntWritable> {
        @Override
        public int getPartition(PairOfStringInt key, IntWritable value, int reducerNo) {
            return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % reducerNo;
        }
    }

    /**
     * Creates an instance of this tool.
     */
    private BuildInvertedIndexCompressed() {
    }

    private static final class Args {
        @Option(name = "-input", metaVar = "[path]", required = true, usage = "input path")
        String input;

        @Option(name = "-output", metaVar = "[path]", required = true, usage = "output path")
        String output;

        @Option(name = "-reducers", metaVar = "[num]", usage = "number of reducers")
        int numReducers = 1;
    }

    /**
     * Runs this tool.
     */
    @Override
    public int run(String[] argv) throws Exception {
        final Args args = new Args();
        CmdLineParser parser = new CmdLineParser(args, ParserProperties.defaults().withUsageWidth(100));

        try {
            parser.parseArgument(argv);
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            return -1;
        }

        LOG.info("Tool name: " + BuildInvertedIndexCompressed.class.getSimpleName());
        LOG.info(" - input path: " + args.input);
        LOG.info(" - output path: " + args.output);
        LOG.info(" - num reducers: " + args.numReducers);

        Configuration conf = getConf();
        Job job = Job.getInstance(conf);
        job.setJobName(BuildInvertedIndexCompressed.class.getSimpleName());
        job.setJarByClass(BuildInvertedIndexCompressed.class);

        job.setNumReduceTasks(args.numReducers);

        FileInputFormat.setInputPaths(job, new Path(args.input));
        FileOutputFormat.setOutputPath(job, new Path(args.output));

        job.setMapOutputKeyClass(PairOfStringInt.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(BytesWritable.class);

        job.setOutputFormatClass(MapFileOutputFormat.class);

        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);
        job.setPartitionerClass(MyPartitioner.class);

        // Delete the output directory if it exists already.
        Path outputDir = new Path(args.output);
        FileSystem.get(conf).delete(outputDir, true);

        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return 0;
    }

    /**
     * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
     *
     * @param args command-line arguments
     * @throws Exception if tool encounters an exception
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BuildInvertedIndexCompressed(), args);
    }
}