package prim;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class PrimMapper extends Mapper<LongWritable, Text, Text, Edge> {

	public HashSet<String> cache = new HashSet<String>();

	/*
	 * function����ÿ��ѡȡ�ĵ�װ�뵽ȫ�ֱ���cache��
	 */
	@Override
	protected void setup(Mapper<LongWritable, Text, Text, Edge>.Context context)
			throws IOException, InterruptedException {
		FileSystem fs = FileSystem.get(context.getConfiguration());
		FileStatus[] fileList = fs.listStatus(new Path(context.getConfiguration().get("selectedNodePath")));
		BufferedReader in = null;
		FSDataInputStream fsi = null;
		String line = null;
		for (int i = 0; i < fileList.length; i++) {
			if (!fileList[i].isDirectory()) {
				fsi = fs.open(fileList[i].getPath());
				in = new BufferedReader(new InputStreamReader(fsi, "UTF-8"));
				while ((line = in.readLine()) != null) {
					String[] arr = line.split("\t");
					for (String node : arr) {
						cache.add(node);
					}
				}
			}
		}
		in.close();
		fsi.close();
	}

	/*
	 * function:ɸѡ����ȫ�ֱ����еĵ��node���������reduce 
	 * parameter0: key ƫ���� 
	 * parameter1��value ����ĵ�ǰ�� 
	 * parameter2: context ������
	 * output: key 1��value ɸѡ���node����
	 */
	protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String[] arr = value.toString().split("\t");
		Edge node = new Edge(arr[0], arr[1], Double.parseDouble(arr[2]));
		Boolean b1 = cache.contains(node.getNode1());
		Boolean b2 = cache.contains(node.getNode2());
		if (b1 ^ b2) {
			context.write(new Text("1"), node);
		}
	}
}
