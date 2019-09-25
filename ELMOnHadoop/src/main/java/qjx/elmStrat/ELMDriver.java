package qjx.elmStrat;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import util.RandomGenerateWeight;
import util.Triple;

/**
 * ELM并行计算隐藏层矩阵H和H.T*H,H.T*T的主类。
 * */
public class ELMDriver {
	public int d;		//数据的维度
	public int l;		//隐层节点的个数
	public int m;		//输出层节点的个数
	public double lambda = 1.0;			//超参数
	public final static Configuration conf = new Configuration();

	public ELMDriver() {
		super();
	}

	public ELMDriver(int d, int l, int m, double lambda) {
		super();
		this.d = d;
		this.l = l;
		this.m = m;
		this.lambda = lambda;
	}

	public void run(String inputWeightFile, String trainingDataFile, String outputWeightFile) {
		FileSystem fs;
		try {
			fs = FileSystem.get(conf);
			RandomGenerateWeight.generate(l, (d + 1), inputWeightFile);
			fs.copyFromLocalFile(new Path(inputWeightFile), new Path("/dan/elm/"));
			conf.setInt("data_dimension", d);
			conf.setInt("hidden_unit", l);
			conf.setInt("class_num", m);
			conf.set("random_weigth", "/dan/elm/"+inputWeightFile");
			Job job = Job.getInstance(conf);

			job.setJarByClass(TestMain.class);

			job.setMapperClass(ELMMapper.class);
			job.setMapOutputKeyClass(Triple.class);
			job.setMapOutputValueClass(DoubleWritable.class);

			job.setCombinerClass(ELMCombiner.class);

			job.setReducerClass(ELMReducer.class);
			job.setOutputKeyClass(Triple.class);
			job.setOutputValueClass(DoubleWritable.class);

			FileInputFormat.setInputPaths(job, new Path(trainingDataFile));
			FileOutputFormat.setOutputPath(job, new Path(outputWeightFile));

			job.waitForCompletion(true);

			double[][] u = new double[l][l];
			double[][] v = new double[l][m];
			double[][] one = new double[l][l];
			for (int i = 0; i < l; i++) {
				for (int j = 0; j < l; j++) {
					if (i == j) {
						one[i][j] = 1.0 / lambda;
					} else {
						one[i][j] = 0.0;
					}
				}
			}
			readFileToArray(conf, outputWeightFile + "/part-r-00000", u, v);
			Array2DRowRealMatrix matrixU = new Array2DRowRealMatrix(u);
			Array2DRowRealMatrix matrixV = new Array2DRowRealMatrix(v);
			Array2DRowRealMatrix matrixOne = new Array2DRowRealMatrix(one);
			RealMatrix betaMatrix = inverseMatrix(matrixOne.add(matrixU)).multiply(matrixV);
			writeArrayToFile("./beta.txt", betaMatrix.getData());
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	//将最终计算出的输出层权值矩阵保存到本地文件中
	public void writeArrayToFile(String localPath, double[][] array) throws Exception {
		FileOutputStream out = new FileOutputStream(localPath);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out));
		for (int i = 0; i < array.length; i++) {
			String line = "";
			for (int j = 0; j < array[i].length; j++) {
				line += array[i][j] + ",";
			}
			bw.write(line);
			bw.write("\r\n");
		}

		bw.close();
	}
	
	//从文件中读取MR计算的矩阵U和矩阵V，path为HDFS路径
	public void readFileToArray(Configuration conf, String path, double[][] u, double[][] v) throws Exception {
		FileSystem fs = FileSystem.newInstance(conf);
		InputStream in = null;
		in = fs.open(new Path(path));
		BufferedReader br = new BufferedReader(new InputStreamReader(in));
		String line = null;
		while ((line = br.readLine()) != null) {
			String[] split = line.split("\t");
			String[] arrIndex = split[0].split(",");
			if (arrIndex[0].equalsIgnoreCase("U")) {
				u[Integer.parseInt(arrIndex[1])][Integer.parseInt(arrIndex[2])] = Double.parseDouble(split[1]);
			} else {
				v[Integer.parseInt(arrIndex[1])][Integer.parseInt(arrIndex[2])] = Double.parseDouble(split[1]);
			}
		}
		br.close();
	}
	
	//计算矩阵的逆
	public RealMatrix inverseMatrix(RealMatrix A) {
		RealMatrix result = new LUDecomposition(A).getSolver().getInverse();
		return result;
	}

}
