package com.original;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class CopyOfClassificationPrediction {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("D://DataScienceCollection//DataSets//Titanic//Titanic_NewDataSet//train.arff");
	
		Instances traindata = source.getDataSet();
		traindata.setClassIndex(traindata.numAttributes()-1);
		int numClasses = traindata.numClasses();
		for (int i=0;i<numClasses;i++){
			String classValue = traindata.classAttribute().value(i);
			System.out.println("the "+i+"th class value:"+classValue);
		}
		System.out.println("------------------------------------------");
		/**
		 * naive bayes classifier	
		 */
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(traindata);
		/**
		 * load test data
		 */
		DataSource source2 = new DataSource("D://DataScienceCollection//DataSets//Titanic//Titanic_NewDataSet//testwithtraindataa.arff");
		Instances testdata = source2.getDataSet();
		testdata.setClassIndex(testdata.numAttributes()-1);
		/**
		 * make prediction by naive bayes classifier
		 */
		int correct = 0;
		int predictions =0;
		for (int j=0;j<testdata.numInstances();j++){
			double actualClass = testdata.instance(j).classValue();
			String actual = testdata.classAttribute().value((int) actualClass);
			Instance newInst = testdata.instance(j);
			double preNB = nb.classifyInstance(newInst);
			String predString = testdata.classAttribute().value((int) preNB);
			System.out.println(actual+","+predString);
			
			if (predString == actual) {
				correct++;
			}
			predictions++;
		}
		
		System.out.print(100 * correct / predictions);
	}

}
