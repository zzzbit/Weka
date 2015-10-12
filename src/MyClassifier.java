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
			 * ����������
			 */
			Classifier classifier1;
			Classifier classifier2;
			Classifier classifier3;
			Classifier classifier4;

			/**
			 * ����ѵ��������������Arff��
			 */
			File inputFile = new File(
					"C:\\Program Files\\Weka-3-6\\data\\glass.arff");// ѵ�������ļ�
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // ����ѵ���ļ�
			
			inputFile = new File(
					"C:\\Program Files\\Weka-3-6\\data\\glass.arff");// ���������ļ�
			atf.setFile(inputFile);
			Instances instancesTest = atf.getDataSet(); // ��������ļ�
			
//			/**
//			 * ����ѵ��������������CSV��
//			 */
//			File inputFile = new File(
//					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\��־��\\�ܽ����\\Weka\\train.csv");// ѵ�������ļ�
//			CSVLoader csv = new CSVLoader();
//			csv.setFile(inputFile);
//			Instances instancesTrain = csv.getDataSet(); // ����ѵ���ļ�
//			
//			inputFile = new File(
//					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\��־��\\�ܽ����\\Weka\\test.csv");// ���������ļ�
//			csv.setFile(inputFile);
//			Instances instancesTest = csv.getDataSet(); // ��������ļ�

			/**
			 * ���÷������������кţ���һ��Ϊ0�ţ���instancesTest.numAttributes()����ȡ����������
			 */
			instancesTest.setClassIndex(0);
			instancesTrain.setClassIndex(0);

			/**
			 * ��ʼ������������
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
			 * ���������ڼ�����ģ�͵���
			 */
			classifier2.buildClassifier(instancesTrain);
			int sum = instancesTest.numInstances();
			int right = 0;
			for (int i = 0; i < sum; i++)// ���Է�����
			{
				if (classifier3.classifyInstance(instancesTest.instance(i)) == instancesTest
						.instance(i).classValue())// ���Ԥ��ֵ�ʹ�ֵ��ȣ����������еķ������ṩ����Ϊ��ȷ�𰸣�����������壩
				{
					right++;// ��ȷֵ��1
				}
			}
			System.out.println("׼ȷ��:" + (right*1.0 / sum));
		} catch (Exception e) {
			// TODO: handle exception
		}
	}
}
