package util;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

public class Triple implements WritableComparable<Triple> {

	
	private String left;
	private Integer middle;
	private Integer right;
	
	public Triple() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Triple(String left, Integer middle, Integer right) {
		super();
		this.left = left;
		this.middle = middle;
		this.right = right;
	}

	public String getLeft() {
		return left;
	}

	public void setLeft(String left) {
		this.left = left;
	}

	public Integer getMiddle() {
		return middle;
	}

	public void setMiddle(Integer middle) {
		this.middle = middle;
	}

	public Integer getRight() {
		return right;
	}

	public void setRight(Integer right) {
		this.right = right;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((left == null) ? 0 : left.hashCode());
		result = prime * result + ((middle == null) ? 0 : middle.hashCode());
		result = prime * result + ((right == null) ? 0 : right.hashCode());
		return result;
	}



	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Triple other = (Triple) obj;
		if (left == null) {
			if (other.left != null)
				return false;
		} else if (!left.equals(other.left))
			return false;
		if (middle == null) {
			if (other.middle != null)
				return false;
		} else if (!middle.equals(other.middle))
			return false;
		if (right == null) {
			if (other.right != null)
				return false;
		} else if (!right.equals(other.right))
			return false;
		return true;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeUTF(this.left);
		out.writeInt(this.middle);
		out.writeInt(this.right);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		this.left = in.readUTF();
		this.middle = in.readInt();
		this.right = in.readInt();
	}

	@Override
	public int compareTo(Triple o) {
		if(this.equals(o)) {
			return 0;
		} 
		if(this.left.compareTo(o.left) < 0) {
			return -1;
		}else if(this.left.compareTo(o.left) == 0) {
			if(this.middle < o.middle) {
				return -1;
			}else if(this.middle == o.middle) {
				if(this.right < o.right) {
					return -1;
				}
			}
		}
		return 1;
	}
	
	@Override
	public String toString() {
		return left + "," + middle + "," + right;
	}


}
