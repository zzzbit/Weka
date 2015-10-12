import java.io.File;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

public class MyClassifier_SavaLoad {

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
					"C:\\Program Files\\Weka-3-6\\data\\breast-cancer.arff");// 训练语料文件
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // 读入训练文件
			
			inputFile = new File(
					"C:\\Program Files\\Weka-3-6\\data\\breast-cancer.arff");// 测试语料文件
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
			 * 6.保存模型文件
			 */
			SerializationHelper.write("LibSVM.model", classifier4);
			SerializationHelper.write("NaiveBayes.model", classifier1);
			SerializationHelper.write("J48.model", classifier2);
			SerializationHelper.write("ZeroR.model", classifier3);
			
			/**
			 * 7.加载模型文件
			 */
			Classifier classifier8 = (Classifier) weka.core.SerializationHelper.read("LibSVM.model");
			Classifier classifier5 = (Classifier) weka.core.SerializationHelper.read("NaiveBayes.model");
			Classifier classifier6 = (Classifier) weka.core.SerializationHelper.read("J48.model");
			Classifier classifier7 = (Classifier) weka.core.SerializationHelper.read("ZeroR.model");
			
			/**
			 * 8.评测加载后的分类器
			 */
			
			Evaluation eval = new Evaluation(instancesTrain);
			
			eval.evaluateModel(classifier8, instancesTest);
			System.out.println(eval.errorRate());
			eval.evaluateModel(classifier5, instancesTest);
			System.out.println(eval.errorRate());
			eval.evaluateModel(classifier6, instancesTest);
			System.out.println(eval.errorRate());
			eval.evaluateModel(classifier7, instancesTest);
			System.out.println(eval.errorRate());
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
