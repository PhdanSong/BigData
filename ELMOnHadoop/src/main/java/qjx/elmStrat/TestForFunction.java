package qjx.elmStrat;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;


/*
 * 测试类
 */
public class TestForFunction {

	public static void main(String[] args) throws Exception {

		String inputWeightFile = "./weight.txt";
		String outputWeightFile = "./beta.txt";
		ELMModel elmModel = new ELMModel(2, 20, 2, 100, inputWeightFile, outputWeightFile);

		String testDataFile = "./t1.txt";
		elmModel.evaluate(testDataFile, 77855);

	}

	// 计算矩阵的逆
	public static RealMatrix inverseMatrix(RealMatrix A) {
		RealMatrix result = new LUDecomposition(A).getSolver().getInverse();
		return result;
	}

}
