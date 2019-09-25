package prim;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class PrimReducer extends Reducer<Text, Edge, Text, Edge> {

	/*
	 * function����map�����node����ѡ��Ȩ����Сnode������� 
	 * parameter0: k2 map������ļ�����1�� 
	 * parameter1:v2 mapɸѡ�������node���� 
	 * parameter2: context ������ 
	 * output:key ��СȨ�ص�node�������� value ��СȨ�ص�node����
	 */
	protected void reduce(Text k2, Iterable<Edge> v2, Context context) throws IOException, InterruptedException {
		if (v2 != null) {
			Edge minNode = getMinNode(v2);
			if (minNode != null) {
				context.write(new Text(minNode.getNode1()), minNode);
			}
		}

	}

	/*
	 * function:�ڰ�������node������ѡ��Ȩ����С��node���� 
	 * parameter��iter ��������node���������
	 * return��������Ȩ����С��node����
	 */
	public Edge getMinNode(Iterable<Edge> iter) {
		Iterator<Edge> iterator = iter.iterator();
		if (!iterator.hasNext()) {
			return null;
		}
		Edge minNode = new Edge(iterator.next());
		while (iterator.hasNext()) {
			Edge nextNode = iterator.next();
			if ((nextNode.getEdge() < minNode.getEdge())) {
				minNode = new Edge(nextNode);
			}
		}
		return minNode;

	}
}
