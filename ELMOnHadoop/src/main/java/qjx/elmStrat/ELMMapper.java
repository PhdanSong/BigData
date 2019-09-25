package qjx.elmStrat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import util.OpUtils;
import util.Triple;
import util.Tuple;

public class ELMMapper extends Mapper<LongWritable, Text, Triple, DoubleWritable> {
	private Integer l = null;		// 隐层节点数
	private Integer m = null;		// 输出层节点数
	private Integer d = null;		// 输入层节点数
	private double[][] u = null;	// H.T*H
	private double[][] v = null;	// H.T*T
	private double[][] w = null;	// 输入层权值矩阵
	private double[] b = null;		// 输入层偏置
	
	
	
	/*
	 * 初始化输入层权值矩阵
	 */
	@Override
	protected void setup(Context context) throws IOException, InterruptedException {
		Configuration conf = context.getConfiguration();
		l = conf.getInt("hidden_unit", 10);
		m = conf.getInt("class_num", 2);
		d = conf.getInt("data_dimension", 2);
		u = new double[l][l];
		v = new double[l][m];
		w = new double[l][d];
		b = new double[l];
		Arrays.fill(b, 0.0);
		Arrays.fill(u, b);
		Arrays.fill(v, b);

		String path = conf.get("random_weigth");
		FileSystem fs = FileSystem.get(conf);
		InputStream in = null;
		try {
			in = fs.open(new Path(path));
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String line = null;
			int row = 0;
			while ((line = br.readLine()) != null) {
				String[] arr = line.split(",");
				w[row] = OpUtils.Array2Double(Arrays.copyOfRange(arr, 0, arr.length - 1));
				b[row] = Double.parseDouble(arr[arr.length - 1]);
				row++;
			}
		} finally {
			IOUtils.closeStream(in);
		}

	}

	@Override
	protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String[] split = value.toString().split(" ");
		Tuple tuple = OpUtils.parse(split);
		double[] x = tuple.getX();
		int t = (int) tuple.getT();
		Double[] oneHot = OpUtils.oneHot(t, m);
		double[] h = new double[l];
		//计算隐层输出
		for (int i = 0; i < l; i++) {
			double z = OpUtils.mutiply(w[i], x) + b[i];
			h[i] = OpUtils.sigmoid(z);
		}
		//计算U和V矩阵
		for (int i = 0; i < l; i++) { 
			for (int j = 0; j < l; j++) {
				u[i][j] = h[i] * h[j];
				context.write(new Triple("U", i, j), new DoubleWritable(u[i][j]));
			}
			
			for (int j = 0; j < m; j++) {
				v[i][j] = h[i] * oneHot[j];
				context.write(new Triple("V", i, j), new DoubleWritable(v[i][j]));
			}
		}

	}

	

}
