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
			 * 1.����������
			 */
			Classifier classifier1;
			Classifier classifier2;
			Classifier classifier3;
			Classifier classifier4;

			/**
			 * 2.����ѵ������������
			 */
			File inputFile = new File(
					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\��־��\\�ܽ����\\Weka\\change_train.arff");// ѵ�������ļ�
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // ����ѵ���ļ�
			
			inputFile = new File(
					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\��־��\\�ܽ����\\Weka\\change_test.arff");// ���������ļ�
			atf.setFile(inputFile);
			Instances instancesTest = atf.getDataSet(); // ��������ļ�

			/**
			 * 3.���÷������������кţ���һ��Ϊ0�ţ���instancesTest.numAttributes()����ȡ����������
			 */
			instancesTest.setClassIndex(0);
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
			 * 5.ѵ��������
			 */
			classifier4.buildClassifier(instancesTrain);
			classifier1.buildClassifier(instancesTrain);
			classifier2.buildClassifier(instancesTrain);
			classifier3.buildClassifier(instancesTrain);
			
			/**
			 * 6.���������
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
