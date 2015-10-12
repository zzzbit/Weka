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
			 * 1.����������
			 */
			Classifier classifier1;
			Classifier classifier2;
			Classifier classifier3;
			Classifier classifier4;

			/**
			 * 2.����������Arff��
			 */
			File inputFile = new File(
					"C:\\Program Files\\Weka-3-6\\data\\breast-cancer.arff");// ѵ�������ļ�
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // ����ѵ���ļ�

			/**
			 * 3.���÷������������кţ���һ��Ϊ0�ţ���instancesTest.numAttributes()����ȡ����������
			 */
			instancesTrain.setClassIndex(0);

			/**
			 * 4.��ʼ������������
			 */
			// ���ر�Ҷ˹�㷨
			classifier1 = (Classifier) Class.forName(
					"weka.classifiers.bayes.NaiveBayes").newInstance();
			// ������
			classifier2 = (Classifier) Class.forName(
					"weka.classifiers.trees.J48").newInstance();
			// Zero
			classifier3 = (Classifier) Class.forName(
					"weka.classifiers.rules.ZeroR").newInstance();
			// LibSVM
			classifier4 = (Classifier) Class.forName(
					"weka.classifiers.functions.LibSVM").newInstance();

			/*
			 * 5.ʮ������֤
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
