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

package ca.uwaterloo.cs451.a1;

import io.bespin.java.util.Tokenizer;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.kohsuke.args4j.ParserProperties;

import tl.lin.data.pair.PairOfFloatInt;
import tl.lin.data.pair.PairOfStrings;

import java.io.IOException;
import java.util.*;
import java.util.HashMap;

import java.io.*;

public class PairsPMI extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(PairsPMI.class);

    private static final class MyMapper extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private static final PairOfStrings PAIR = new PairOfStrings();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            List<String> tokens = Tokenizer.tokenize(value.toString());

            int wordEnd = Math.min(40, tokens.size());
            Set<String> words = new TreeSet<String>();

            for (int i = 0; i < wordEnd; i++) {
                words.add(tokens.get(i));
            }

            for (String wordA : words) {
                for (String wordB : words) {
                    if (!wordA.equals(wordB)) {
                        PAIR.set(wordA, wordB);
                        context.write(PAIR, ONE);
                    }
                }
            }
        }
    }

    private static final class MyCombiner extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
        private static final IntWritable SUM = new IntWritable();

        @Override
        public void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            Iterator<IntWritable> iter = values.iterator();
            while (iter.hasNext()) {
                sum += iter.next().get();
            }
            SUM.set(sum);
            context.write(key, SUM);
        }
    }

    private static final class MyReducer extends Reducer<PairOfStrings, IntWritable, PairOfStrings, PairOfFloatInt> {
        private static HashMap<String, Integer> wordCounts = new HashMap<String, Integer>();
        private static int lineCount = 0;

        private static final PairOfFloatInt VALUE = new PairOfFloatInt();
        private int threshold = 40;

        @Override
        public void setup(Context context) throws IOException, InterruptedException {

            threshold = context.getConfiguration().getInt("threshold", 40);

            Path pt = new Path("temp/part-r-00000");
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(pt)));

            // set line count
            String line = br.readLine();
            String[] parts = line.split("\\s+");
            int count = Integer.parseInt(parts[1]);
            lineCount = count;

            // set word count
            while ((line = br.readLine()) != null) {
                parts = line.split("\\s+");
                String word = parts[0];
                count = Integer.parseInt(parts[1]);
                wordCounts.put(word, count);
            }
            br.close();
        }

        @Override
        public void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            Iterator<IntWritable> iter = values.iterator();
            while (iter.hasNext()) {
                sum += iter.next().get();
            }
            if (sum < threshold)
                return;
            int countA = wordCounts.get(key.getLeftElement());
            int countB = wordCounts.get(key.getRightElement());

            double tmp = (sum * lineCount * 1.0d) / (countA * countB);
            double pmi = Math.log10(tmp);

            VALUE.set((float) pmi, sum);
            context.write(key, VALUE);
        }
    }

    // Mapper: emits (token, 1) for each different word.
    // Mapper: emits (*, 1) for line count
    public static final class CountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        // Reuse objects to save overhead of object creation.
        private static final IntWritable ONE = new IntWritable(1);
        private static final Text WORD = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            List<String> tokens = Tokenizer.tokenize(value.toString());
            int end = Math.min(tokens.size(), 40);

            Set<String> words = new TreeSet<String>();

            for (int i = 0; i < end; i++) {
                words.add(tokens.get(i));
            }

            for (String word : words) {
                WORD.set(word);
                context.write(WORD, ONE);
            }

            WORD.set("*");
            context.write(WORD, ONE);
        }
    }

    public static final class CountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        // Reuse objects.
        private static final IntWritable SUM = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            // Sum up values.
            Iterator<IntWritable> iter = values.iterator();
            int sum = 0;
            while (iter.hasNext()) {
                sum += iter.next().get();
            }
            SUM.set(sum);
            context.write(key, SUM);
        }
    }

    /**
     * Creates an instance of this tool.
     */
    private PairsPMI() {
    }

    private static final class Args {
        @Option(name = "-input", metaVar = "[path]", required = true, usage = "input path")
        String input;

        @Option(name = "-output", metaVar = "[path]", required = true, usage = "output path")
        String output;

        @Option(name = "-reducers", metaVar = "[num]", usage = "number of reducers")
        int numReducers = 1;

        @Option(name = "-threshold", metaVar = "[num]", usage = "threshold of pairs")
        int threshold = 10;
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

        LOG.info("Tool name: " + PairsPMI.class.getSimpleName());
        LOG.info(" - input path: " + args.input);
        LOG.info(" - output path: " + args.output);
        LOG.info(" - num reducers: " + args.numReducers);
        LOG.info(" - threshold: " + args.threshold);

        // Delete the output directory if it exists already.
        Path tmp = new Path("temp");
        Path outputDir = new Path(args.output);
        FileSystem.get(getConf()).delete(outputDir, true);
        FileSystem.get(getConf()).delete(tmp, true);

        // Mark start time
        long startTime = System.currentTimeMillis();

        // Line Count Job
        Job job1 = Job.getInstance(getConf());
        job1.setJobName("Counter");
        job1.setJarByClass(PairsPMI.class);
        FileInputFormat.setInputPaths(job1, new Path(args.input));
        FileOutputFormat.setOutputPath(job1, new Path("temp"));

        job1.setMapperClass(CountMapper.class);
        job1.setReducerClass(CountReducer.class);
        job1.setCombinerClass(CountReducer.class);

        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(IntWritable.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);
        job1.setOutputFormatClass(TextOutputFormat.class);

        job1.getConfiguration().setInt("mapred.max.split.size", 1024 * 1024 * 32);
        job1.getConfiguration().set("mapreduce.map.memory.mb", "3072");
        job1.getConfiguration().set("mapreduce.map.java.opts", "-Xmx3072m");
        job1.getConfiguration().set("mapreduce.reduce.memory.mb", "3072");
        job1.getConfiguration().set("mapreduce.reduce.java.opts", "-Xmx3072m");

        if (!job1.waitForCompletion(true)) {
            System.exit(1);
        }

        Job job2 = Job.getInstance(getConf());
        job2.setJobName(PairsPMI.class.getSimpleName());
        job2.setJarByClass(PairsPMI.class);

        job2.setNumReduceTasks(args.numReducers);

        job2.getConfiguration().setInt("threshold", args.threshold);

        FileInputFormat.setInputPaths(job2, new Path(args.input));
        FileOutputFormat.setOutputPath(job2, new Path(args.output));

        job2.setMapOutputKeyClass(PairOfStrings.class);
        job2.setMapOutputValueClass(IntWritable.class);
        job2.setOutputKeyClass(PairOfStrings.class);
        job2.setOutputValueClass(PairOfFloatInt.class);

        job2.setMapperClass(MyMapper.class);
        job2.setCombinerClass(MyCombiner.class);
        job2.setReducerClass(MyReducer.class);

        job2.getConfiguration().setInt("mapred.max.split.size", 1024 * 1024 * 32);
        job2.getConfiguration().set("mapreduce.map.memory.mb", "3072");
        job2.getConfiguration().set("mapreduce.map.java.opts", "-Xmx3072m");
        job2.getConfiguration().set("mapreduce.reduce.memory.mb", "3072");
        job2.getConfiguration().set("mapreduce.reduce.java.opts", "-Xmx3072m");

        job2.waitForCompletion(true);
        System.out.println("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return 0;
    }

    /**
     * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
     *
     * @param args command-line arguments
     * @throws Exception if tool encounters an exception
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new PairsPMI(), args);
    }
}