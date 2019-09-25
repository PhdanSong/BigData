import org.apache.spark.{SparkContext,SparkConf }
object wordCountReal {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf //创建SparkConf对象
    conf.setAppName("wordCount") //设置应用程序的名称，在程序运行的监控界面可以看到名称
    conf.setMaster("local") //此时，程序在本地运行，不需要安装Spark集群
    val sc = new SparkContext(conf)
    val lines = sc.textFile("A://data.txt", 1)//
    val words = lines.flatMap { line => line.split(",") } //对每一行的字符串进行单词拆分并把所有行的拆分结果通过flat合并成为
    val pairs = words.map { word => (word, 1) }
    val wordCounts = pairs.reduceByKey(_+_) //对相同的key，进行value的累计
    wordCounts.foreach(map => println(map._1 +":"+ map._2))

    sc.stop()
  }
}
