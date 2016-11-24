/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclusterer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Visat
 */
public class MyKMeans extends RandomizableClusterer implements
  NumberOfClustersRequestable {
    
    protected Instances instances;
    protected Instances centroids;
    protected List<Instance>[] clusters;    
    protected int K = 3;
    protected int iterations;
    protected int maxIterations = 500;
    protected final DistanceFunction distanceFunction = new EuclideanDistance();                  
    
    @Override
    public void buildClusterer(Instances instances) throws Exception {        
        int N = instances.numInstances();
        if (K < 1) K = 1;
        if (N == 0 || N < K) return;
        getCapabilities().testWithFail(instances);
        
        this.instances = instances;        
        distanceFunction.setInstances(instances);        
                 
        // assign first centroids randomly
        Random rand = new Random();        
        Set<Integer> centroidIdx = new HashSet<>();        
        while (centroidIdx.size() < K) {
            int x = rand.nextInt(N);
            centroidIdx.add(x);
        }                                                  
        centroids = new Instances(instances, K);        
        centroidIdx.forEach((idx) -> {                        
            centroids.add(instances.instance(idx));
        });                       
        
        int[] prevCluster = new int[N];
        for (int i = 0; i < N; ++i) prevCluster[i] = -1;
        
        List<Integer>[] tmpCluster = new List[K];
        for (int i = 0; i < K; ++i) tmpCluster[i] = new ArrayList<>();
        
        boolean converged = false;
        iterations = 0;        
        while (!converged && iterations < maxIterations) {         
            ++iterations;
            converged = true;            

            for (int i = 0; i < K; ++i) tmpCluster[i].clear();
            for (int i = 0; i < N; ++i) {
                int cluster = clusterInstance(instances.instance(i));
                if (prevCluster[i] != cluster) {
                    converged = false;
                    prevCluster[i] = cluster;
                }
                tmpCluster[cluster].add(i);                
            }
            
            // update centroid
            centroids = new Instances(instances, K);
            for (int i = 0; i < K; ++i) {
                Instances members = new Instances(instances, N);                
                for (Integer member: tmpCluster[i])
                    members.add(instances.instance(member));                                
                centroids.add(createCentroid(members));
            }
        }        
        clusters = new List[K];
        for (int i = 0; i < K; ++i) {
            clusters[i] = new ArrayList<>();
            for (Integer member: tmpCluster[i])
                clusters[i].add(instances.instance(member));
        }
    }

    @Override
    public int numberOfClusters() throws Exception {
        return K;
    }
    
    @Override
    public void setNumClusters(int K) throws Exception {
        if (K <= 0) throw new Exception("Number of clusters must be > 0");
        this.K = K;
    }
    
    public int getIterations(){
        return iterations;
    }
    
    public int getMaxIterations() {
        return maxIterations;
    }
    
    public void setMaxIterations(int maxIterations) throws Exception {
        if (maxIterations <= 0) throw new Exception("Number of iterations must be > 0");
        this.maxIterations = maxIterations;
    }
    
    @Override
    public Capabilities getCapabilities() {        
      Capabilities result = super.getCapabilities();
      result.disableAll();
      result.enable(Capability.NO_CLASS);

      // attributes
      result.enable(Capability.NOMINAL_ATTRIBUTES);
      result.enable(Capability.NUMERIC_ATTRIBUTES);
      result.enable(Capability.MISSING_VALUES);

      return result;
    }
    
    private Instance createCentroid(Instances members) {
        double[] vals = new double[members.numAttributes()];
        double[][] nominalDists = new double[members.numAttributes()][];
        double[] weightMissing = new double[members.numAttributes()];
        double[] weightNonMissing = new double[members.numAttributes()];
                
        for (int j = 0; j < members.numAttributes(); j++) {
            if (members.attribute(j).isNominal()) {
                nominalDists[j] = new double[members.attribute(j).numValues()];
            }
        }
        for (int i = 0; i < members.numInstances(); ++i) {
            Instance inst = members.instance(i);
            for (int j = 0; j < members.numAttributes(); j++) {
                if (inst.isMissing(j)) {
                    weightMissing[j] += inst.weight(); 
                }
                else {
                    weightNonMissing[j] += inst.weight();
                    if (members.attribute(j).isNumeric())
                        vals[j] += inst.weight() * inst.value(j);
                    else
                        nominalDists[j][(int)inst.value(j)] += inst.weight();                    
                }
            }      
        }
        for (int i = 0; i < members.numAttributes(); i++) {
            if (members.attribute(i).isNumeric()) {
                if  (weightNonMissing[i] > 0) {
                    vals[i] /= weightNonMissing[i];
                } else {
                    vals[i] = Instance.missingValue();
                }
            }
            else {
                double max = -Double.MAX_VALUE;
                double maxIndex = -1;
                for (int j = 0; j < nominalDists[i].length; j++) {
                    if (nominalDists[i][j] > max) {
                        max = nominalDists[i][j];
                        maxIndex = j;
                    }
                    vals[i] = max < weightMissing[i] ? Instance.missingValue() : maxIndex;                    
                }
            }
        }
        return new Instance(1.0, vals);
    }
             
    @Override
    public int clusterInstance(Instance instance) throws Exception {
        double min = Double.MAX_VALUE;
        int idx = 0;
        for (int i = 0; i < K; ++i) {
            double dist = distanceFunction.distance(centroids.instance(i), instance);
            if (dist < min) {
                min = dist;
                idx = i;
            }
        }
        return idx;
    }

    @Override
    public String toString() {
        if (centroids == null) {
            return "No clusterer built yet!";
        }

        int maxWidth = 0;
        int maxAttWidth = 0;
        boolean containsNumeric = false;
        for (int i = 0; i < K; i++) {
            for (int j = 0 ;j < centroids.numAttributes(); j++) {
                if (centroids.attribute(j).name().length() > maxAttWidth) {
                    maxAttWidth = centroids.attribute(j).name().length();
                }
                if (centroids.attribute(j).isNumeric()) {
                    containsNumeric = true;
                    double width = Math.log(Math.abs(centroids.instance(i).value(j))) / Math.log(10.0);
                    if (width < 0) {
                        width = 1;
                    }
                    // decimal + # decimal places + 1
                    width += 6.0;
                    if ((int)width > maxWidth) {
                        maxWidth = (int)width;
                    }
                }
            }
        }

        for (int i = 0; i < centroids.numAttributes(); i++) {
            if (centroids.attribute(i).isNominal()) {
                Attribute a = centroids.attribute(i);
                for (int j = 0; j < centroids.numInstances(); j++) {
                    String val = a.value((int)centroids.instance(j).value(i));
                    if (val.length() > maxWidth) {
                        maxWidth = val.length();
                    }
                }
                for (int j = 0; j < a.numValues(); j++) {
                    String val = a.value(j) + " ";
                    if (val.length() > maxAttWidth) {
                        maxAttWidth = val.length();
                    }
                }
            }
        }

        // check for size of cluster sizes
        for (int i = 0; i < clusters.length; i++) {
            String size = "(" + clusters[i].size() + ")";
            if (size.length() > maxWidth) {
                maxWidth = size.length();
            }
        }

        String plusMinus = "+/-";
        maxAttWidth += 2;
        if (maxAttWidth < "Attribute".length() + 2) {
            maxAttWidth = "Attribute".length() + 2;
        }

        if (maxWidth < "Full Data".length()) {
            maxWidth = "Full Data".length() + 1;
        }

        if (maxWidth < "missing".length()) {
            maxWidth = "missing".length() + 1;
        }



        StringBuffer temp = new StringBuffer();
        //    String naString = "N/A";


    /*    for (int i = 0; i < maxWidth+2; i++) {
          naString += " ";
          } */
        temp.append("\nkMeans\n======\n");
        temp.append("\nNumber of iterations: " + iterations+"\n");

        /*if(distanceFunction instanceof EuclideanDistance){
            temp.append("Within cluster sum of squared errors: " + Utils.sum(squaredErrors));
        }else{
            temp.append("Sum of within cluster distances: " + Utils.sum(squaredErrors));
        }*/

        temp.append("\n\nCluster centroids:\n");
        temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2)) - "Cluster#".length(), true));

        temp.append("\n");
        temp.append(pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));


//        temp.append(pad("Full Data", " ", maxWidth + 1 - "Full Data".length(), true));

        // cluster numbers
        for (int i = 0; i < K; i++) {
            String clustNum = "" + i;
            temp.append(pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
        }
        temp.append("\n");

        // cluster sizes
        String cSize = "";
        temp.append(pad(cSize, " ", maxAttWidth - cSize.length(), true));
        for (int i = 0; i < K; i++) {
            cSize = "(" + clusters[i].size() + ")";
            temp.append(pad(cSize, " ",maxWidth + 1 - cSize.length(), true));
        }
        temp.append("\n");

        temp.append(pad("", "=", maxAttWidth +
                (maxWidth * (centroids.numInstances())
                        + centroids.numInstances()), true));
        temp.append("\n");

        for (int i = 0; i < centroids.numAttributes(); i++) {
            String attName = centroids.attribute(i).name();
            temp.append(attName);
            for (int j = 0; j < maxAttWidth - attName.length(); j++) {
                temp.append(" ");
            }

            String strVal;
            String valMeanMode;
            for (int j = 0; j < K; j++) {
                if (centroids.attribute(i).isNominal()) {
                    if (centroids.instance(j).isMissing(i)) {
                        valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode =
                                pad((strVal = centroids.attribute(i).value((int)centroids.instance(j).value(i))),
                                        " ", maxWidth + 1 - strVal.length(), true);
                    }
                } else {
                    if (centroids.instance(j).isMissing(i)) {
                        valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode = pad((strVal = Utils.doubleToString(centroids.instance(j).value(i),
                                maxWidth,4).trim()),
                                " ", maxWidth + 1 - strVal.length(), true);
                    }
                }
                temp.append(valMeanMode);
            }
            temp.append("\n");
        }

        temp.append("\n\n");
        return temp.toString();
    }

    private String pad(String source, String padChar,
                       int length, boolean leftPad) {
        StringBuffer temp = new StringBuffer();

        if (leftPad) {
            for (int i = 0; i< length; i++) {
                temp.append(padChar);
            }
            temp.append(source);
        } else {
            temp.append(source);
            for (int i = 0; i< length; i++) {
                temp.append(padChar);
            }
        }
        return temp.toString();
    }
    
    public static void main(String[] args) {
        runClusterer(new MyKMeans(), args);
    }    
}
