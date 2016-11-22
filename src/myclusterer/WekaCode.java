/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclusterer;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.DensityBasedClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author mochamadtry
 */
public class WekaCode {
    
    public static final int SimpleKMeans = 0;
    public static final int MyKMeans = 1;
    public static final int MyAgnes = 2;
    private Clusterer clusterer;
    private static ClusterEvaluation eval;
    
    public static Instances readFileArff(String fileName) throws Exception{
        //http://weka.sourceforge.net/doc.stable/weka/core/Instances.html
        //membaca semua instances dari file .arff, .csv
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(fileName);
        Instances dataSet = source.getDataSet();
        //set atribut terakhir sebagai kelas 
        //if (dataSet.classIndex()== -1)
        //     dataSet.setClassIndex(dataSet.numAttributes() - 1); //Make the last attribute be the class
        return dataSet; 
    }
    
    public static Instances removeAttributes(Instances data, String attribute) throws Exception{
        Remove remove = new Remove(); 
        remove.setAttributeIndices(attribute); //Set which attributes are to be deleted (or kept if invert is true)
        remove.setInputFormat(data); //Sets the format of the input instances.
        Instances filterData = Filter.useFilter(data, remove); //Filters an entire set of instances through a filter and returns the new set.
        return filterData;    
    }
    
    //Filter : resample 
    public static Instances resampleData(Instances data) throws Exception{
        Resample resample = new Resample(); 
        resample.setInputFormat(data);
        Instances filterData = Filter.useFilter(data, resample);
        return filterData; 
    }
    
    //Build Cluster
    public static Clusterer buildClusterer(Instances dataSet, int clusterType) throws Exception{
        Clusterer clusterer = null;
        if (clusterType == SimpleKMeans){
            clusterer = (SimpleKMeans) new SimpleKMeans();
            clusterer.buildClusterer(dataSet);
        }
        else if(clusterType == MyKMeans){
            MyKMeans kmeans = new MyKMeans();
            Scanner scan = new Scanner(System.in);
            int K = scan.nextInt();
            kmeans.setNumberOfClusters(K);
            clusterer = kmeans;
            clusterer.buildClusterer(dataSet);
        }
        else if(clusterType == MyAgnes){
            clusterer = new MyAgnes();
            clusterer.buildClusterer(dataSet);
        }
        return clusterer;
    }
    
    public void evaluateModel (Instances dataTest) throws Exception {
        eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        eval.evaluateClusterer(dataTest);
    }
    
    //Lihat semua data 
    public static Instances classifyUnseenData (Clusterer clusterer, Instances dataSet) throws Exception{
        Instances labeledData = new Instances(dataSet);
        // labeling data
        for (int i = 0; i < labeledData.numInstances(); i++) {
            double clsLabel = clusterer.clusterInstance(dataSet.instance(i));
            labeledData.instance(i).setClassValue(clsLabel);
        }
        return labeledData;
    }
    
    //Using model to classify one unseen data(input data)
    public static void classifyUnseenData(String[] attributes, Clusterer clusterer, Instances data) throws Exception {
        Instance newInstance = new Instance(data.numAttributes());
        newInstance.setDataset(data);
        for (int i = 0; i < data.numAttributes()-1; i++) {
            if(Attribute.NUMERIC == data.attribute(i).type()){
                Double value = Double.valueOf(attributes[i]);
                newInstance.setValue(i, value);
            } else {
                newInstance.setValue(i, attributes[i]);
            }
        }
        
        double clsLabel = clusterer.clusterInstance(newInstance);
        newInstance.setClassValue(clsLabel);
        
        String result = data.classAttribute().value((int) clsLabel);
        
        System.out.println("Hasil Classify Unseen Data Adalah: " + result);
    }
    
    public static ClusterEvaluation getEval(){
        return eval;
    }
}
