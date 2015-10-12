import java.io.File;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

public class MyClassifier {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			/**
			 * 声明分类器
			 */
			Classifier classifier1;
			Classifier classifier2;
			Classifier classifier3;
			Classifier classifier4;

			/**
			 * 读入训练、测试样本（Arff）
			 */
			File inputFile = new File(
					"C:\\Program Files\\Weka-3-6\\data\\glass.arff");// 训练语料文件
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // 读入训练文件
			
			inputFile = new File(
					"C:\\Program Files\\Weka-3-6\\data\\glass.arff");// 测试语料文件
			atf.setFile(inputFile);
			Instances instancesTest = atf.getDataSet(); // 读入测试文件
			
//			/**
//			 * 读入训练、测试样本（CSV）
//			 */
//			File inputFile = new File(
//					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\张志智\\总结积累\\Weka\\train.csv");// 训练语料文件
//			CSVLoader csv = new CSVLoader();
//			csv.setFile(inputFile);
//			Instances instancesTrain = csv.getDataSet(); // 读入训练文件
//			
//			inputFile = new File(
//					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\张志智\\总结积累\\Weka\\test.csv");// 测试语料文件
//			csv.setFile(inputFile);
//			Instances instancesTest = csv.getDataSet(); // 读入测试文件

			/**
			 * 设置分类属性所在行号（第一行为0号），instancesTest.numAttributes()可以取得属性总数
			 */
			instancesTest.setClassIndex(0);
			instancesTrain.setClassIndex(0);

			/**
			 * 初始化分类器类型
			 */
			// 朴素贝叶斯算法
			classifier1 = (Classifier) Class.forName(
					"weka.classifiers.bayes.NaiveBayes").newInstance();
			// 决策树
			classifier2 = (Classifier) Class.forName(
					"weka.classifiers.trees.J48").newInstance();
			// Zero
			classifier3 = (Classifier) Class.forName(
					"weka.classifiers.rules.ZeroR").newInstance();
			// LibSVM
			classifier4 = (Classifier) Class.forName(
					"weka.classifiers.functions.LibSVM").newInstance();

			/*
			 * 即它是用于检测分类模型的类
			 */
			classifier2.buildClassifier(instancesTrain);
			int sum = instancesTest.numInstances();
			int right = 0;
			for (int i = 0; i < sum; i++)// 测试分类结果
			{
				if (classifier3.classifyInstance(instancesTest.instance(i)) == instancesTest
						.instance(i).classValue())// 如果预测值和答案值相等（测试语料中的分类列提供的须为正确答案，结果才有意义）
				{
					right++;// 正确值加1
				}
			}
			System.out.println("准确率:" + (right*1.0 / sum));
		} catch (Exception e) {
			// TODO: handle exception
		}
	}
}
