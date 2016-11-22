/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclusterer;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
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
        
        //Baca input file 
        Scanner scan = new Scanner(System.in); 
        nameOfFile= scan.nextLine(); 
        try {
            //Baca File arff
            dataSet = WekaCode.readFileArff(nameOfFile);
            System.out.println(dataSet.firstInstance());
            
            //Build Clusterer
            System.out.println("Tuliskan model classifier : 0.SimpleKMeans / 1.MyKMeans / 2.MyAgnes ");
            int clustererType = scan.nextInt();
            clusterer = WekaCode.buildClusterer(dataSet, clustererType);
            System.out.println(clusterer.getClass());
            System.out.println("-----");
            System.out.println();
            System.out.println("Number of iterations: "+);
            
            //Given test set 
        } catch (Exception ex) {
            Logger.getLogger(MyClusterer.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}