package util;

import static org.junit.Assert.assertSame;

import java.util.Arrays;

public class OpUtils {

	public OpUtils() {
		super();
	}

	public static Double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public static Double mutiply(double[] w, double[] x) {
		assertSame("array length is mismatch!!", w.length, x.length);
		double mutiply = 0.0;
		for (int i = 0; i < w.length; i++) {
			mutiply += w[i] * x[i];
		}
		return mutiply;
	}

	public static double[] Array2Double(String[] arr) {
		if (arr == null) {
			return null;
		}
		double[] d = new double[arr.length];
		for (int i = 0; i < d.length; i++) {
			d[i] = Double.parseDouble(arr[i]);
		}
		return d;
	}
	public static Tuple parse(String[] split) {
		double[] x = new double[split.length - 1];
		double t = Double.parseDouble(split[split.length - 1]);
		for (int i = 0; i < x.length; i++) {
			x[i] = Double.parseDouble(split[i]);
		}
		return new Tuple(x, t);
	}
	
	public static Double[] oneHot(int t, int classNum) {
		Double[] onehot = new Double[classNum];
		Arrays.fill(onehot, 0.0);
		onehot[t - 1] = 1.0; 
		return onehot;
	}

}


