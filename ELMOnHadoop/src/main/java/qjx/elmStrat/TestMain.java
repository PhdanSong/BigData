package qjx.elmStrat;

public class TestMain {
	public static void main(String[] args) throws Exception {
		int d = Integer.parseInt(args[0]);
		int l = Integer.parseInt(args[1]);
		int m = Integer.parseInt(args[2]);
		double lambda = Double.parseDouble(args[3]);

		ELMModel model = new ELMModel(d, l, m, lambda, args[4], args[5]);
		model.training(args[6]);

	}

}
