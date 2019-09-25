import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class RandomGenerateWeight {

	public static void generate(int row, int col, double mean, double variance, String path) throws IOException {
		Random random = new Random();
		random.setSeed(1L);
		FileWriter out = null;
		out = new FileWriter(new File(path));
		for (int l = 0; l < row; l++) {
			Double[] unit = new Double[col];
			for (int j = 0; j < col; j++) {
				unit[j] = Math.sqrt(variance) * random.nextGaussian() + mean; // 高斯分布
			}

			for (int i = 0; i < unit.length; i++) {
				out.write(String.format("%.5f", unit[i]) + " ");
			}
			out.write("\r\n"); 

		}
		out.close();

	}
}
