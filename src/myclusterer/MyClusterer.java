/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclusterer;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.core.Instances;

/**
 *
 * @author Scemo
 */
public class MyClusterer {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        
        String nameOfFile; 
        Clusterer clusterer; 
        Instances dataSet;
        ClusterEvaluation eval;
        
        //Baca input file 
        Scanner scan = new Scanner(System.in); 
        System.out.print("Masukkan nama file untuk di-cluster: ");
        nameOfFile= scan.nextLine(); 
        try {
            //Baca File arff
            dataSet = WekaCode.readFileArff(nameOfFile);
            
            //Build Clusterer
            System.out.println("Tuliskan model clusterer : 0.SimpleKMeans / 1.HierarchicalClusterer / 2.MyKMeans / 3.MyAgnes ");
            int clustererType = scan.nextInt();
            clusterer = WekaCode.buildClusterer(dataSet, clustererType);
            eval = new ClusterEvaluation();
            eval.setClusterer(clusterer);
            eval.evaluateClusterer(dataSet);
            System.out.println("Cluster Evaluation:");
            System.out.println(eval.clusterResultsToString());
            
            //Given test set 
        } catch (Exception ex) {
            Logger.getLogger(MyClusterer.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
