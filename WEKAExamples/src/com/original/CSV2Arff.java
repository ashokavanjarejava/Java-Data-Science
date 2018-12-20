package com.original;
import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;



public class CSV2Arff {
	public static void main(String[] args) throws IOException{
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("D://DataScienceCollection//DataSets//Titanic//Titanic_NewDataSet//test.csv"));
		//loader.setSource(new File("D:\\DataScienceCollection\\DataSets\\titanic_test.csv"));
		Instances data = loader.getDataSet();
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File("D://DataScienceCollection//DataSets//Titanic//Titanic_NewDataSet//test.arff"));
		saver.writeBatch();
		System.out.print("Activity Completed");
		
	}
}
