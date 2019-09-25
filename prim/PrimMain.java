package prim;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class PrimMain {

	/*
	 * function:mapReduce������
	 * parameter0:args[0]������С����������ʼ��,�����һ����ΪselectedNode.txt���ļ���ÿ�ε����󣬻����ѡ��ĵ���µ�����ļ���
	 * parameter1:args[1] �����ļ�����ͨͼ��ÿһ�м�¼�ĸ�ʽ�����+tab��+�յ�+tab��+Ȩ��
	 * parameter2:args[2]��������ļ���·�������ѡ��ı߻���outputPath�б��� 
	 * parameter3:args[3] ��ͨͼ�ĵ�ĸ���
	 */
	public static void main(String[] args) throws Exception {

		if (args.length < 4) {
			System.err.println("usage: <selectedNodePath, inputPath, outputPath, nodeNumber>");
			return;
		}

		Configuration conf = new Configuration();
		conf.set("selectedNodePath", args[0]);

		String inputPath = args[1];
		String outputPath = args[2];
		int nodeNumber = Integer.parseInt(args[3]);
		FileSystem fs = FileSystem.get(conf);
		while (nodeNumber > 0) {
			conf.setInt("nodeNumber", nodeNumber);
			Job job = Job.getInstance(conf);

			job.setJarByClass(PrimMain.class);

			job.setNumReduceTasks(1); // set number of ReduceTask is 1

			job.setMapperClass(PrimMapper.class);
			job.setMapOutputKeyClass(Text.class);
			job.setMapOutputValueClass(Edge.class);

			job.setReducerClass(PrimReducer.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Edge.class);

			FileInputFormat.setInputPaths(job, new Path(inputPath));
			FileOutputFormat.setOutputPath(job, new Path(outputPath + "/" + nodeNumber + "/"));

			job.waitForCompletion(true);
			nodeNumber--;
			updateSelectNode(nodeNumber, outputPath, conf, fs);

		}
		combineAllPartToOne(Integer.parseInt(args[3]), outputPath, conf, fs);

	}

	/*
	 * function:��ÿ��ѡ��ĵ�ŵ�selectedNode.txt�� 
	 * parameter0��nodeNumber ��ͨͼ�ĵ���
	 * parameter1��ÿ��ִ��mapreduce���������ڵ��ļ��� 
	 * parameter2: conf �����ļ�
	 * parameter3: fs �ļ�ϵͳ
	 */
	private static void updateSelectNode(int nodeNumber, String outputPath, Configuration conf, FileSystem fs)
			throws Exception {
		FSDataInputStream in = null;
		FSDataOutputStream out = null;
		try {
			in = fs.open(new Path(outputPath + "/" + (nodeNumber + 1) + "/part-r-00000"));
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			out = fs.append(new Path(conf.get("selectedNodePath") + "/selectedNode.txt"));
			String line = null;
			while ((line = br.readLine()) != null) {
				String[] selectedNodes = line.split("\t");
				for (String selectedNode : selectedNodes) {
					out.writeBytes(selectedNode);
					out.writeBytes("\r");
				}
			}
		} finally {
			IOUtils.closeStream(in);
			IOUtils.closeStream(out);
		}
	}

	/*
	 * function:�����ѡ��ĵ�Ͷ�Ӧ��Ȩ�غϲ���һ���ļ��� 
	 * parameter0��nodeNumber ��ͨͼ�ĵ���
	 * parameter1�� ÿ��ִ��mapreduce���������ڵ��ļ���
	 * parameter2: conf �����ļ�
	 * parameter3: fs �ļ�ϵͳ
	 */
	private static void combineAllPartToOne(int nodeNumber, String outputPath, Configuration conf, FileSystem fs)
			throws Exception {
		while (nodeNumber > 1) {
			FSDataInputStream in = null;
			FSDataOutputStream out = null;
			try {
				in = fs.open(new Path(outputPath + "/" + nodeNumber + "/part-r-00000"));
				BufferedReader br = new BufferedReader(new InputStreamReader(in));
				out = fs.append(new Path(outputPath + "/" + 1 + "/part-r-00000"));
				String line = null;
				while ((line = br.readLine()) != null) {
					String[] Nodes = line.split("\t");
					out.writeBytes(Nodes[1]);
					out.writeBytes("\t");
					out.writeBytes(Nodes[2]);
					out.writeBytes("\t");
					out.writeBytes(Nodes[3]);
					out.writeBytes("\t");
					out.writeBytes("\r");
				}
			} finally {
				IOUtils.closeStream(in);
				IOUtils.closeStream(out);
			}
			nodeNumber--;
		}
	}

}
