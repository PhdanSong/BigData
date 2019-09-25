package demo

import org.apache.spark.{SparkConf, SparkContext}

object LocalWordCount {
  def main(args: Array[String]): Unit = {
    //配置Spark的任务
    val conf = new SparkConf().setAppName("MyScalaWordCount").setMaster("local")

    //创建一个SparkContext对象
    val sc = new SparkContext(conf)

    //读入数据
    val lines = sc.textFile("d:\\temp\\data.txt")

    //分词
    val words = lines.flatMap(_.split(" "));

    //每个单词记一次数
    val wordPair = words.map((_,1)) //完整写法： words.map(x => (x,1))

    //使用reduce进行统计
    val wordcount = wordPair.reduceByKey(_+_)

    //调用一个action，触发计算
    val result = wordcount.collect()

    //打印在屏幕上
    result.foreach(println)

    //停止SparkContext
    sc.stop()
  }
}
