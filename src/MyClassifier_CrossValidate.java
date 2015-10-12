import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

public class MyClassifier_CrossValidate {

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
			 * 2.读入样本（Arff）
			 */
			File inputFile = new File(
					"C:\\Program Files\\Weka-3-6\\data\\breast-cancer.arff");// 训练语料文件
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // 读入训练文件

			/**
			 * 3.设置分类属性所在行号（第一行为0号），instancesTest.numAttributes()可以取得属性总数
			 */
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
			 * 5.十交叉验证
			 */
			Evaluation eval = new Evaluation(instancesTrain);
			eval.crossValidateModel(classifier4, instancesTrain, 10, new Random(1));
			System.out.println(eval.errorRate());
			eval.crossValidateModel(classifier1, instancesTrain, 10, new Random(1));
			System.out.println(eval.errorRate());
			eval.crossValidateModel(classifier2, instancesTrain, 10, new Random(1));
			System.out.println(eval.errorRate());
			eval.crossValidateModel(classifier3, instancesTrain, 10, new Random(1));
			System.out.println(eval.errorRate());
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
