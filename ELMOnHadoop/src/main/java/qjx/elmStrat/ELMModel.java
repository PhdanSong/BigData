package qjx.elmStrat;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import util.OpUtils;
import util.Tuple;

/*
 * ELM模型的主类，包含训练、评估、预测三个方法。
 */
public class ELMModel {
	public int d;
	public int l;
	public int m;
	public double lambda;
	public String inputWeightFile;
	public String outputWeightFile;

	public ELMModel() {
		super();
	}

	public ELMModel(int d, int l, int m, double lambda, String inputWeightFile, String outputWeightFile) {
		super();
		this.d = d;
		this.l = l;
		this.m = m;
		this.lambda = lambda;
		this.inputWeightFile = inputWeightFile;
		this.outputWeightFile = outputWeightFile;
	}

	// 训练模型
	public void training(String trainingDataFile) {
		ELMDriver elmDriver = new ELMDriver(d, l, m, lambda);
		elmDriver.run(inputWeightFile, trainingDataFile, outputWeightFile);
	}

	// 预测
	@SuppressWarnings("unused")
	public double[] predict(double[][] x) {
		double[][] xAddOneColunm = new double[x.length][d + 1];
		for (int i = 0; i < xAddOneColunm.length; i++) {
			for (int j = 0; j < xAddOneColunm[i].length; j++) {
				if (j == xAddOneColunm[i].length - 1) {
					xAddOneColunm[i][j] = 1.0;
				} else {
					xAddOneColunm[i][j] = x[i][j];
				}
			}
		}
		double[] pred = new double[x.length];
		
		Array2DRowRealMatrix data = (Array2DRowRealMatrix) new Array2DRowRealMatrix(xAddOneColunm).transpose(); //d+1 * n
		Array2DRowRealMatrix inputWeight = readInputWeightFile(inputWeightFile); // l * (d + 1)
		Array2DRowRealMatrix outputWeight = (Array2DRowRealMatrix) readOutputWeightFile(outputWeightFile).transpose(); // m*l
	
		double[][] z = inputWeight.multiply(data).getData();
		for (int i = 0; i < z.length; i++) {
			for (int j = 0; j < z[i].length; j++) {
				z[i][j] = OpUtils.sigmoid(z[i][j]);
			}
			// System.out.println(Arrays.toString(z[i]));
		}
		Array2DRowRealMatrix hiddenMatrix = new Array2DRowRealMatrix(z);
		Array2DRowRealMatrix predMatrix = (Array2DRowRealMatrix) outputWeight.multiply(hiddenMatrix).transpose(); // n
																													// *m

		double[][] predArr = predMatrix.getData();
		for (int i = 0; i < predArr.length; i++) {
			double max = predArr[i][0];
			int maxIndex = 0;
			for (int j = 1; j < predArr[i].length; j++) {
				if (predArr[i][j] > max) {
					max = predArr[i][j];
					maxIndex = j;
				}
			}
			pred[i] = maxIndex + 1; // 类别是索引值加1
		}
//		System.out.println("pred is: " + Arrays.toString(predArr[1500]));
		return pred;
	}

	// 从本地读取ELM的输出层权值矩阵
	public Array2DRowRealMatrix readOutputWeightFile(String outputWeightFile) {
		double[][] outputWeight = new double[l][m];
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(outputWeightFile));
			String line = null;
			int row = 0;
			while ((line = br.readLine()) != null) {
				String[] split = line.split(",");
				outputWeight[row] = OpUtils.Array2Double(split);
				row++;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		try {
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return new Array2DRowRealMatrix(outputWeight);
	}

	// 从本地读取ELM的输入层权值矩阵
	public Array2DRowRealMatrix readInputWeightFile(String inputWeightFile) {
		double[][] inputWeight = new double[l][d + 1];
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(inputWeightFile));
			String line = null;
			int row = 0;
			while ((line = br.readLine()) != null) {
				String[] split = line.split(",");
				inputWeight[row] = OpUtils.Array2Double(split);
				row++;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		try {
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return new Array2DRowRealMatrix(inputWeight);

	}

	// 评估模型
	@SuppressWarnings("unused")
	public void evaluate(String testDataFile, int dataNum) {
		double[][] testData = new double[dataNum][d];
		double[] tTestReal = new double[dataNum];
		double[] tTestPred = new double[dataNum];
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(testDataFile));
			String line = null;
			int row = 0;
			while ((line = br.readLine()) != null) {
				String[] split = line.split(" ");
				Tuple parse = OpUtils.parse(split);
				double[] xTest = parse.getX();
				double t = parse.getT();
				testData[row] = xTest;
				tTestReal[row] = t;
				row++;
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		tTestPred = predict(testData);
		int rightNum = 0;
		for (int i = 0; i < tTestPred.length; i++) {
			if (tTestPred[i] == tTestReal[i]) {
				rightNum++;
			}
		}
		System.out.println("right num is:" + rightNum);
		// System.out.println("真实类别：" + Arrays.toString(Arrays.copyOfRange(tTestReal, 0,
		// 20)));
		// System.out.println("预测类别：" + Arrays.toString(Arrays.copyOfRange(tTestPred, 0,
		// 20)));
		System.out.println("accuracy is:" + (rightNum * 1.0 / dataNum));

	}

}
