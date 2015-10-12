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

public class MyClassifier_Arff {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			/**
			 * 1.声明分类器
			 */
			Classifier classifier1;
			Classifier classifier2;
			Classifier classifier3;
			Classifier classifier4;

			/**
			 * 2.读入训练、测试样本
			 */
			File inputFile = new File(
					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\张志智\\总结积累\\Weka\\change_train.arff");// 训练语料文件
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // 读入训练文件
			
			inputFile = new File(
					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\张志智\\总结积累\\Weka\\change_test.arff");// 测试语料文件
			atf.setFile(inputFile);
			Instances instancesTest = atf.getDataSet(); // 读入测试文件

			/**
			 * 3.设置分类属性所在行号（第一行为0号），instancesTest.numAttributes()可以取得属性总数
			 */
			instancesTest.setClassIndex(0);
			instancesTrain.setClassIndex(0);

			/**
			 * 4.初始化分类器类型
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
			 * 5.训练分类器
			 */
			classifier4.buildClassifier(instancesTrain);
			classifier1.buildClassifier(instancesTrain);
			classifier2.buildClassifier(instancesTrain);
			classifier3.buildClassifier(instancesTrain);
			
			/**
			 * 6.评测分类器
			 */
			
			Evaluation eval = new Evaluation(instancesTrain);
			
			eval.evaluateModel(classifier4, instancesTest);
			System.out.println(eval.errorRate());
			eval.evaluateModel(classifier1, instancesTest);
			System.out.println(eval.errorRate());
			eval.evaluateModel(classifier2, instancesTest);
			System.out.println(eval.errorRate());
			eval.evaluateModel(classifier3, instancesTest);
			System.out.println(eval.errorRate());
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
