import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.util.Collections;
import java.util.Comparator;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

public class KNN
{

    public static class Dis_Label {
        public float dis;//距离
        public String label;//标签
    }

    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {
        private ArrayList<ArrayList<Float>> test = new ArrayList<ArrayList<Float>> ();

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException
        {
            //key是训练数据行号
            context.setStatus(key.toString());
            String[] s = value.toString().split(",");
            String label = s[s.length - 1];
            for (int i=0; i<test.size(); i++){
                ArrayList<Float> curr_test = test.get(i);
                double tmp = 0;
                for(int j=0; j<curr_test.size(); j++){
                    tmp += (curr_test.get(j) - Float.parseFloat(s[j]))*(curr_test.get(j) - Float.parseFloat(s[j]));
                }
                context.write(new Text(Integer.toString(i)), new Text(Double.toString(tmp)+","+label)); //测试样例编号,所有训练集距离&标签                 
            }

        }
        protected void setup(org.apache.hadoop.mapreduce.Mapper<Object, Text, Text, Text>.Context context) throws java.io.IOException, InterruptedException {
            // load the test vectors
            FileSystem fs = FileSystem.get(context.getConfiguration());
            Configuration conf = context.getConfiguration();;
            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(context.getConfiguration().get(
                    "iris.test", new String(conf.get("test")))))));
            String line = br.readLine();
            int count = 0;
            while (line != null) {
                String[] s = line.split(",");
                ArrayList<Float> testcase = new ArrayList<Float>();
                for (int i = 0; i < s.length-1; i++){
                    testcase.add(Float.parseFloat(s[i]));
                }
                test.add(testcase);
                line = br.readLine();
                count++;
            }
            br.close();
        }
    }

    public static class KNNCombiner extends Reducer<Text, Text, Text, Text>
    {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException
        {
            ArrayList<Dis_Label> dis_Label_set = new ArrayList<Dis_Label>();
            for (Text value : values){
                String[] s = value.toString().split(","); //拆开所有 距离+标签
                Dis_Label tmp = new Dis_Label();
                tmp.label = s[1];
                tmp.dis = Float.parseFloat(s[0]);
                dis_Label_set.add(tmp);
            }
            //排序 
            Collections.sort(dis_Label_set, new Comparator<Dis_Label>(){
                @Override
                public int compare(Dis_Label a, Dis_Label b){
                    if (a.dis > b.dis){
                        return 1; //小的在前
                    }
                    return -1;
                }
            });

            Configuration conf = context.getConfiguration();
            final int k = new Integer(conf.get("k")); //K值

            //统计前K个最近样例的标签
            for (int i=0; i<dis_Label_set.size() && i<k; i++){
                context.write(key, new Text(dis_Label_set.get(i).label));
            }
        }
    }

    public static class KNNReducer extends Reducer<Text, Text, Text, Text>
    {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException
        {
             HashMap<String, Integer> ans = new HashMap<String, Integer>();
             for(Text val:values)
             {
                 if (!ans.containsKey(val)){
                     ans.put(val.toString(), 0);
                 }
                 ans.put(val.toString(), ans.get(val.toString())+1);
             }
             //确定标签
             int mx = -1;
             String ansLabel = "";
             for (String l:ans.keySet()){
                 if (mx < ans.get(l)){
                     mx = ans.get(l);
                     ansLabel = l;
                 }
             }
             context.write(key, new Text(ansLabel));
        }
    }

    public static void main(String[] args) throws Exception
    {
        Configuration conf=new Configuration();
        if(args.length!=4)
        {
            System.out.println("Usage: KNN <train> <test> <out> <k>");
            System.exit(2);
        }
        Path inputPath=new Path(args[0]);
        Path outputPath=new Path(args[2]);
        outputPath.getFileSystem(conf).delete(outputPath, true);

        conf.set("test",args[1]);
        conf.set("k",args[3]);

        Job job=Job.getInstance(conf, "KNN");
        job.setJarByClass(KNN.class);

        job.setMapperClass(TokenizerMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setCombinerClass(KNNCombiner.class);

        job.setReducerClass(KNNReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, outputPath);

        System.exit(job.waitForCompletion(true)? 0:1);
    }
}  