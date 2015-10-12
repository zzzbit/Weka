import java.io.File;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffLoader;

public class Ensemble {

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
			Classifier[] cfsArray = new Classifier[4];

			/**
			 * ����ѵ������������
			 */
			File inputFile = new File(
					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\��־��\\�ܽ����\\Weka\\train.arff");// ѵ�������ļ�
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // ����ѵ���ļ�
			
			inputFile = new File(
					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\��־��\\�ܽ����\\Weka\\test.arff");// ���������ļ�
			atf.setFile(inputFile);
			Instances instancesTest = atf.getDataSet(); // ��������ļ�

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

			/**
			 * ����
			 */
			cfsArray[0] = classifier1;
			cfsArray[1] = classifier2;
			cfsArray[2] = classifier3;
			cfsArray[3] = classifier4;
			Vote ensemble = new Vote();
			/*
			 * ����ensemble�������ľ��߷�ʽ��Ҫ�У� AVERAGE_RULE PRODUCT_RULE
			 * MAJORITY_VOTING_RULE MIN_RULE MAX_RULE MEDIAN_RULE
			 * ���Ǿ���Ĺ�����ʽ����ҿ��Բο�weka��˵���ĵ��� ����������ѡ����Ƕ���ͶƱ�ľ��߹���
			 */
			SelectedTag tag1 = new SelectedTag(Vote.MAJORITY_VOTING_RULE,
					Vote.TAGS_RULES);
			ensemble.setCombinationRule(tag1);
			ensemble.setClassifiers(cfsArray);
			// �������������
			ensemble.setSeed(2);
			// ѵ��ensemble������
			ensemble.buildClassifier(instancesTrain);

			/*
			 * ���������ڼ�����ģ�͵���
			 */
			Evaluation testingEvaluation = new Evaluation(instancesTest);
			int length = instancesTest.numInstances();
			Instance testInst;
			for (int i = 0; i < length; i++) {
				testInst = instancesTest.instance(i);
				// ͨ�������������ÿ�������������Է�������Ч��
				testingEvaluation.evaluateModelOnceAndRecordPrediction(
						ensemble, testInst);

			}
			System.out
					.println("����������ȷ�ʣ�" + (1 - testingEvaluation.errorRate()));
		} catch (Exception e) {
			// TODO: handle exception
		}
	}

}
