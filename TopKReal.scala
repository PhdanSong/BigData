import org.apache.spark.{SparkContext,SparkConf }
/*
* 功能：实现top-K
* 步骤：读入磁盘文件，以逗号为分隔符，拆分数据，然后映射为两元素的数组，再降序，然后取每个元素组的第一个元素，take是取前几个元素
* 性质：保留，正确
* 日期：2018.7.19
* */
object TopKReal {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf //创建SparkConf对象
    conf.setAppName("wordCount") //设置应用程序的名称，在程序运行的监控界面可以看到名称
    conf.setMaster("local") //此时，程序在本地运行，不需要安装Spark集群
    val sc = new SparkContext(conf)
    val lines = sc.textFile("A://topKData.txt", 1)//
    val words = lines.flatMap { line => line.split(",") } //对每一行的字符串进行单词拆分并把所有行的拆分结果通过flat合并成为一
//  words.collect()//collect用于打印一个弹性分布式数据集元素
    println("输入的数据是")
    words.foreach(print)
    println("前K个数据是")
    words.map(m=>(m.toInt,m.toInt)).sortByKey(false).map(m=>m._1).take(3).foreach(println _)//ascending=false降序
    print("map处理后的数据是")
    val mapRDD=words.map(m=>(m.toInt,m.toInt))
      mapRDD.foreach(println)
    print("sortByKey处理后的数据是")
    val sortByKeyRDD=mapRDD.sortByKey(false)
    sortByKeyRDD.foreach(println)
  }
}
