package com.exercise;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class SupervisedLinearRegressionPrediction {
	public static void main(String[] args) throws Exception{
		DataSource trainingData = new DataSource("C:\\MyFolder\\WEKA\\WEKAExamples\\DataFiles\\house.arff");
		DataSource testingData = new DataSource("C:\\MyFolder\\WEKA\\WEKAExamples\\DataFiles\\house-unknown.arff");
		
		
		Instances trainingDataSet = trainingData.getDataSet();
		trainingDataSet.setClassIndex(trainingDataSet.numAttributes()-1);		
		Instances testingDataSet = testingData.getDataSet();
		testingDataSet.setClassIndex(testingDataSet.numAttributes()-1);
		
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(trainingDataSet);
				
		String formula = String.valueOf(lr);
		System.out.println("PRINTING THE FORMULA");
		System.out.println(formula);
		//NOW EXTRACT THE VALUES FROM UNKNOWN FILE INTO THE FORMULA TO PREDICT THE VALUE
		
		//975,2947,5,1,1,?
		
		/*price = 
			  195.2035 * size +
			  38.9694 * land +
			  76218.4642 * granite +
			  73947.2118 * extra_bathroom +
			   2681.136
		 */ 
		
		//(195.2035 * 975) + (38.9694 * 2947) + (76218.4642 * 1) + (73947.2118 * 1) + 2681.136
		
		//458013.0463
		
		
		//--------------------------------------------------------OR----------------------------------------------------
		
		
		
		Evaluation eval = new Evaluation(trainingDataSet);
		eval.evaluateModel(lr, testingDataSet);
		double value = lr.classifyInstance(testingDataSet.lastInstance());	
		System.out.println(value);
		
	    //458013.16703945777
	}
}
