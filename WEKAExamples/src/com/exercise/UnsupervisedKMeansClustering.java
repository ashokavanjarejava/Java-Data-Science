package com.exercise;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class UnsupervisedKMeansClustering {
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("C:\\MyFolder\\MOUSE\\WEKA\\WEKAExamples\\DataFiles\\weather.arff");
		Instances traindata = source.getDataSet();
		//traindata.setClassIndex(traindata.numAttributes()-1);
		
		SimpleKMeans kmeans = new SimpleKMeans();
		kmeans.setNumClusters(4);
		kmeans.buildClusterer(traindata);
		
		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(kmeans);
		eval.evaluateClusterer(traindata);
		System.out.println(eval.clusterResultsToString());
	}
}
