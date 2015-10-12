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
			 * 1.声明分类器
			 */
			Classifier classifier1;
			Classifier classifier2;
			Classifier classifier3;
			Classifier classifier4;
			Classifier[] cfsArray = new Classifier[4];

			/**
			 * 2.读入训练、测试样本
			 */
			File inputFile = new File(
					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\张志智\\课程\\计算机视觉\\大作业\\实验数据\\Feature\\trainSetHOG.arff");// 训练语料文件
			ArffLoader atf = new ArffLoader();
			atf.setFile(inputFile);
			Instances instancesTrain = atf.getDataSet(); // 读入训练文件
			
			inputFile = new File(
					"C:\\Users\\zhangzhizhi\\Documents\\Everyone\\张志智\\课程\\计算机视觉\\大作业\\实验数据\\Feature\\testSetHOG.arff");// 测试语料文件
			atf.setFile(inputFile);
			Instances instancesTest = atf.getDataSet(); // 读入测试文件

			/**
			 * 3.设置分类属性所在行号（第一行为0号），instancesTest.numAttributes()可以取得属性总数
			 */
			instancesTest.setClassIndex(144);
			instancesTrain.setClassIndex(144);

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
			
			/**
			 * 集成
			 */
			cfsArray[0] = classifier1;
			cfsArray[1] = classifier2;
			cfsArray[2] = classifier3;
			cfsArray[3] = classifier4;
			Vote ensemble = new Vote();
			/*
			 * 订制ensemble分类器的决策方式主要有： AVERAGE_RULE PRODUCT_RULE
			 * MAJORITY_VOTING_RULE MIN_RULE MAX_RULE MEDIAN_RULE
			 * 它们具体的工作方式，大家可以参考weka的说明文档。 在这里我们选择的是多数投票的决策规则
			 */
			SelectedTag tag1 = new SelectedTag(Vote.MAJORITY_VOTING_RULE,
					Vote.TAGS_RULES);
			ensemble.setCombinationRule(tag1);
			ensemble.setClassifiers(cfsArray);
			// 设置随机数种子
			ensemble.setSeed(2);
			// 训练ensemble分类器
			ensemble.buildClassifier(instancesTrain);

			/*
			 * 即它是用于检测分类模型的类
			 */
			Evaluation testingEvaluation = new Evaluation(instancesTest);
			int length = instancesTest.numInstances();
			Instance testInst;
			for (int i = 0; i < length; i++) {
				testInst = instancesTest.instance(i);
				// 通过这个方法来用每个测试样本测试分类器的效果
				testingEvaluation.evaluateModelOnceAndRecordPrediction(
						ensemble, testInst);

			}
			System.out.println("分类器的正确率：" + (1 - testingEvaluation.errorRate()));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
