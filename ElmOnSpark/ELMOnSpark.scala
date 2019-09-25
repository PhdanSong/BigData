import org.apache.hadoop.io.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import java.io.BufferedInputStream
import java.io.FileInputStream
import java.net.URI
import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}

/*
  * parameter0:基本路径
  * parameter1:原始数据的存放路径
  * parameter2:类别数
  * parameter3:样例维度数
  * parameter4:样例数
  * parameter5:隐层节点个数
  * */
object ELMOnSpark {

  def main(args: Array[String]) {

    //  基本路径
    var base_path = args(0)
    //  原始数据的存放路径
    var  dataPath= args(1)
    //  原始数据中的属性处理后的存放路径
    var attrbutesPath = base_path + "attributes//"
    //  原始数据中的类别oneHot编码后的存放路径
    var labelPath = base_path + "labels//"
    //  类别数
    var num_class = args(2).toInt
    //  权重矩阵的保存路径
    var weightPath = base_path
    //  样例维度数+1
    var dataDim = args(3).toInt
    //  样例数
    var dataNum = args(4).toInt
    //  类标索引（样例的最后一列）
    var label_index =args(3).toInt-1
    //  隐层节点个数
    var hiddenNodeNum: Int = args(5).toInt

    val conf = new SparkConf().setAppName("ELMOnSpark")
    val sc = new SparkContext(conf)

    //  处理数据
    val data = sc.textFile(dataPath)
    val dataRDD = data.map(line => {
      val array = line.split(" ")
      val attribute = new Array[Double](array.length)
      for (i <- 0 until array.length-1) {
        attribute(i) = array(i).toDouble
      }
      attribute(array.length-1)=1
      attribute
    }).map(line => Vectors.dense(line))
    val X = new RowMatrix(dataRDD)

    //  对类别oneHot编码
    val data1 = sc.textFile(dataPath)
    data1.map(x => x.split(" ")).map(arr => oneHotEncoder(arr(label_index), num_class)).saveAsTextFile(labelPath)

    //  初始化输入层权重
    RandomGenerateWeight.generate(dataDim, hiddenNodeNum, 0, 1, "Ld.txt")

    //  将权重矩阵上传至hdfs上
    val in = new BufferedInputStream(new FileInputStream("Ld.txt"))
    val conf1 =  new Configuration()
    val fs = FileSystem.get(URI.create(weightPath + "//Ld"), conf1)
    val out = fs.create(new Path(weightPath + "//Ld"))
    IOUtils.copyBytes(in, out, 4096, true)

    //  将权重文件读成矩阵
    val rdd1 = sc.textFile(weightPath + "//Ld")
      .flatMap(_.split(" ")
        .map(_.toDouble)).collect()
    val W_b = Matrices.dense(dataDim, hiddenNodeNum, rdd1)
    sc.parallelize(W_b.rowIter.map(_.toArray.mkString(" ")).toList).repartition(1).saveAsTextFile(weightPath + "//W")

    //  计算隐层输出矩阵H
    val Z = X.multiply(W_b)
    val G = Z.rows.map(x => x.toArray.map(sigmoid)).map(Vectors.dense(_))
    val H = new RowMatrix(G)    // num_data, hiddenNodeNum

    //  保存隐层输出
    H.rows.map(_.toArray.mkString(" ")).repartition(1)
      .saveAsTextFile(base_path + "H")

    //  U的计算：H的转置*H
    val U = H.computeGramianMatrix()
    sc.parallelize(U.rowIter.map(_.toArray.mkString(" ")).toBuffer).saveAsTextFile(base_path + "U" )

    //  V的计算：H的转置*T
    val rdd2 = sc.textFile(labelPath + "part-*", hiddenNodeNum)
      .flatMap(_.split(" ").map(_.toDouble)).collect()
    val T_com=Matrices.dense(num_class,dataNum,rdd2)
    val T=T_com.transpose
    sc.parallelize(T.rowIter.map(_.toArray.mkString(" ")).toList).saveAsTextFile(base_path+"T")
    val H_T=transpose(base_path, base_path + "H//part-*",sc,args(5).toInt)
    val V=H_T.multiply(T)
    V.rows.map(_.toArray.mkString(" "))
      .saveAsTextFile(base_path + "V")

  }
  //  Compute H transpose
  def transpose(basePath :String, HPath:String,sc:SparkContext,numPartition:Int) :RowMatrix={
    val H_ = sc.textFile(HPath, numPartition.toInt)
      .zipWithIndex()
      .map(line => line._2 + " " + line._1.split(" ").mkString(" "))
      .map(_.split(" "))
      .map(fun)
      .filter(_ != "")
      .repartition(1)
      .saveAsTextFile(basePath + "pH")

    val rdd = sc.textFile(basePath + "pH/part-*" , numPartition.toInt)
      .filter(!_.isEmpty)
      .map(_.split(" ").map(_.toDouble))
      .map(arr =>new MatrixEntry(arr(0).toLong,arr(1).toLong, arr(2)))
    val H = new CoordinateMatrix(rdd)
    val H_T = H.transpose().toRowMatrix()
    H_T
  }

  //  Data pre-process
  def fun(arr: Array[String]) ={
    var lines = ""
    for(i <- 1 until arr.length) {
      lines = lines + arr(0)  + " " + (i - 1) + " " +  arr(i) + "\n"
    }
    lines
  }

  //  OneHot
  def oneHotEncoder(label:String, n_classes: Int) ={
    val oneHot = new Array[Int](n_classes)
    for(i <- 0 until  oneHot.length){
      if (i == label.toDouble - 1){
        oneHot(i) = 1
      }else{
        oneHot(i) = 0
      }
    }
    oneHot.mkString(" ")
  }

  def sigmoid(element:Double):Double ={
    1 / (1 + math.exp(-element))
  }
}
