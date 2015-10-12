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

public class Code {

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
			Classifier[] cfsArray = new Classifier[4];

			/**
			 * 2.����ѵ������������
			 */
			File inputFile = new File(
					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\��־��\\�γ�\\������Ӿ�\\����ҵ\\ʵ������\\Feature\\trainSetHOG.arff");// ѵ�������ļ�
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // ����ѵ���ļ�
			
			inputFile = new File(
					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\��־��\\�γ�\\������Ӿ�\\����ҵ\\ʵ������\\Feature\\testSetHOG.arff");// ���������ļ�
			atf.setFile(inputFile);
			Instances instancesTest = atf.getDataSet(); // ��������ļ�

			/**
			 * 3.���÷������������кţ���һ��Ϊ0�ţ���instancesTest.numAttributes()����ȡ����������
			 */
			instancesTest.setClassIndex(144);
			instancesTrain.setClassIndex(144);

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
			System.out.println("����������ȷ�ʣ�" + (1 - testingEvaluation.errorRate()));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
