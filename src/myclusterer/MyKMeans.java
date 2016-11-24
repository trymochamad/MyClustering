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
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

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
    
    public static void main(String[] args) {
        runClusterer(new MyKMeans(), args);
    }    
}
