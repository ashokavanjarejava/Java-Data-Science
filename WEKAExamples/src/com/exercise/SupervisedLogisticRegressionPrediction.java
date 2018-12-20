package com.exercise;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class SupervisedLogisticRegressionPrediction {
	public static void main(String[] args) throws Exception{
		DataSource trainingData = new DataSource("C:\\MyFolder\\MOUSE\\WEKA\\WEKAExamples\\DataFiles\\weather.arff");
		DataSource testingData = new DataSource("C:\\MyFolder\\MOUSE\\WEKA\\WEKAExamples\\DataFiles\\weather-unknown.arff");
		
		
		Instances trainingDataSet = trainingData.getDataSet();
		trainingDataSet.setClassIndex(trainingDataSet.numAttributes()-1);		
		Instances testingDataSet = testingData.getDataSet();
		testingDataSet.setClassIndex(testingDataSet.numAttributes()-1);
		
		Classifier lr = new weka.classifiers.functions.Logistic();
		lr.buildClassifier(trainingDataSet);
				
		
		System.out.println(lr);

			
		Evaluation eval = new Evaluation(trainingDataSet);
		eval.evaluateModel(lr, testingDataSet);
		double value = lr.classifyInstance(testingDataSet.lastInstance());	
		System.out.println(value);
		
	 
	}
}
