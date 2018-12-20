package com.original;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class CrossValidate {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("C:\\MyFolder\\MOUSE\\WEKA\\WEKAExamples\\DataFiles\\iris-demo.arff");
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		NaiveBayes nb = new NaiveBayes();
		int fold = 10;
		int seed = 1;
		Random rand = new Random(seed);
		Instances randData = new Instances(dataset);
		randData.randomize(rand);
		if (randData.classAttribute().isNominal())
			randData.stratify(fold);
		double averagecorrect = 0;
		for (int n=0;n<fold;n++){
			Evaluation eval = new Evaluation(randData);
			Instances train = randData.trainCV(fold, n);
			System.out.println(train);
			Instances test = randData.testCV(fold, n);
			System.out.println(test);
			nb.buildClassifier(train);
			eval.evaluateModel(nb, test);
			double correct = eval.pctCorrect();
			averagecorrect = averagecorrect + correct;
			System.out.println("the "+n+"th cross validation:"+eval.toSummaryString());
			
		}
		System.out.println("the average correction rate of "+fold+" cross validation: "+averagecorrect/fold);
	}
}


/*

ORIGINAL FILE

4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
5.1,3.8,1.9,0.4,Iris-setosa
4.8,3.0,1.4,0.3,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
7.2,3.6,6.1,2.5,Iris-virginica
6.5,3.2,5.1,2.0,Iris-virginica
6.4,2.7,5.3,1.9,Iris-virginica
6.8,3.0,5.5,2.1,Iris-virginica
5.7,2.5,5.0,2.0,Iris-virginica
5.8,2.8,5.1,2.4,Iris-virginica
6.2,3.4,5.4,2.3,Iris-virginica
5.9,3.0,5.1,1.8,Iris-virginica
5.0,2.3,3.3,1.0,Iris-versicolor
5.6,2.7,4.2,1.3,Iris-versicolor*/





/*4.7,3.2,1.3,0.2,Iris-setosa==================2
4.6,3.1,1.5,0.2,Iris-setosa=====================================5
5.0,3.6,1.4,0.2,Iris-setosa============1
5.4,3.9,1.7,0.4,Iris-setosa==============================4
5.1,3.8,1.9,0.4,Iris-setosa==========================3
4.8,3.0,1.4,0.3,Iris-setosa============================================6
7.0,3.2,4.7,1.4,Iris-versicolor========1
6.4,3.2,4.5,1.5,Iris-versicolor==============================================================9
6.9,3.1,4.9,1.5,Iris-versicolor=====================================================8
5.5,2.3,4.0,1.3,Iris-versicolor=======================================================================10
5.0,2.3,3.3,1.0,Iris-versicolor==============2
5.6,2.7,4.2,1.3,Iris-versicolor===============================================7
7.2,3.6,6.1,2.5,Iris-virginica===============================================================9
6.5,3.2,5.1,2.0,Iris-virginica======================================================8
6.4,2.7,5.3,1.9,Iris-virginica=======================3
6.8,3.0,5.5,2.1,Iris-virginica========================================================================10
5.7,2.5,5.0,2.0,Iris-virginica==================================5
5.8,2.8,5.1,2.4,Iris-virginica=========================================6
6.2,3.4,5.4,2.3,Iris-virginica================================================7
5.9,3.0,5.1,1.8,Iris-virginica==========================4*/



