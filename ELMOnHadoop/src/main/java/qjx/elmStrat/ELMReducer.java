package qjx.elmStrat;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Reducer;

import util.Triple;

public class ELMReducer extends Reducer<Triple, DoubleWritable, Triple, DoubleWritable> {

	@Override
	protected void reduce(Triple triple, Iterable<DoubleWritable> values,Context context)
			throws IOException, InterruptedException {
		Double sum = 0.0;
		for(DoubleWritable value : values) {
			sum += value.get();
		}
		context.write(new Triple(triple.getLeft(), triple.getMiddle(), triple.getRight()), new DoubleWritable(sum));
	}

}
