package com.exercise;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;

public class SupervisedClassificationPrediction {
	public static void main(String args[]) throws Exception{
		PrintStream o = new PrintStream(new File("C:\\MyFolder\\iris_output.txt"));	
		//Load the TRAINING DataSet
		DataSource source = new DataSource("C:\\MyFolder\\iris.arff");
		//TRAINING DataSet Loaded
		Instances trainDataset = source.getDataSet();	
		//Set class index to the last attribute which is “@attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}”
		//This Means that - 0=sepal_length
		//1=sepal_width
		//2=petal_length
		//3=petal_width
		//4=The class of flower it will be any of Iris-setosa,Iris-versicolor or Iris-virginica.
		
		trainDataset.setClassIndex(trainDataset.numAttributes()-1);

		//Build model	
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(trainDataset);
		//output model  - As we have 4 attributes —>  sepal_length | sepal_width | petal_length | petal_width. Therefore we get the brief data outcome for all these.
		System.setOut(o);
		System.out.println(nb);

		/*	
		Naive Bayes Classifier

		                         Class
		Attribute          Iris-setosa Iris-versicolor  Iris-virginica
		                        (0.32)          (0.34)          (0.34)
		===============================================================
		sepallength
		  mean                   4.9919          5.9379          6.5795
		  std. dev.              0.3617          0.5042          0.6353
		  weight sum                 48              50              50
		  precision              0.1059          0.1059          0.1059
		
		sepalwidth
		  mean                   3.4091          2.7687          2.9629
		  std. dev.              0.3949          0.3038          0.3088
		  weight sum                 48              50              50
		  precision              0.1091          0.1091          0.1091
		
		petallength
		  mean                   1.4721          4.2452          5.5516
		  std. dev.              0.1813          0.4712          0.5529
		  weight sum                 48              50              50
		  precision              0.1405          0.1405          0.1405
		
		petalwidth
		  mean                   0.2762          1.3097          2.0343
		  std. dev.              0.1115          0.1915          0.2646
		  weight sum                 48              50              50
		  precision              0.1143          0.1143          0.1143
		*/

		//Now, load the TEST DataSet
		DataSource source1 = new DataSource("C:\\MyFolder\\iris-unknown.arff");
		//TEST DataSet Loaded
		Instances testDataset = source1.getDataSet();
		/*
		@relation iris

		@attribute sepallength numeric
		@attribute sepalwidth numeric
		@attribute petallength numeric
		@attribute petalwidth numeric
		@attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}
		
		@data
		5.1,3.5,1.4,0.2,?
		4.9,3,1.4,0.2,?
		7.3,2.9,6.3,1.8,?
		*/

		//Set class index to the last attribute which is “@attribute class {Iris-setosa,Iris-versicolor,Iris-virginica}”
		//This Means that - 0=sepal_length
		//1=sepal_width
		//2=petal_length
		//3=petal_width
		//4=The class of flower it will be any of Iris-setosa,Iris-versicolor or Iris-virginica.
		testDataset.setClassIndex(testDataset.numAttributes()-1);

		//Loop through the TEST DataSet to make predictions
		System.setOut(o);
		System.out.println("========================");
		System.out.println("Actual Class, SMO Predicted");
			 
		// As we have only 3 record therefore, testDataset.numInstances() = 3. Loop will iterate 3 times.
		for (int i = 0; i < testDataset.numInstances(); i++) {
			
			double actualValue = testDataset.instance(i).classValue();
			
			String actual = testDataset.classAttribute().value((int) actualValue);
			System.setOut(o);
			System.out.println("actual  >"+actual); 
			Instance newInst = testDataset.instance(i);		
			double predSMO = nb.classifyInstance(newInst);			
			String predString = testDataset.classAttribute().value((int) predSMO);	
			System.setOut(o);
			System.out.println("predString  –> "+predString);
			

		}
		
		
		
	
	}
}