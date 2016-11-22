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
import weka.clusterers.AbstractClusterer;
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
public class MyKMeans extends AbstractClusterer {
    
    protected Instances instances;
    protected Instances centroids;
    protected List<Instance>[] clusters;    
    protected int K = 3;
    protected int iterations = 0;
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
                int cluster = cluster(instances.instance(i));
                if (prevCluster[i] != cluster) {
                    converged = false;
                    prevCluster[i] = cluster;
                }
                tmpCluster[cluster].add(i);                
            }
            
            // update centroid
            centroids = new Instances(instances, K);
            for (int i = 0; i < K; ++i) {
                Instances tmpInstances = new Instances(instances, 0);                
                for (Integer member: tmpCluster[i])
                    tmpInstances.add(instances.instance(member));
                double[] vals = new double[tmpCluster[i].size()];
                for (int j = 0; j < tmpCluster[i].size(); ++j) {
                    vals[j] = tmpInstances.meanOrMode(j);
                }
                centroids.add(new Instance(1.0, vals));
            }
        }        
        for (int i = 0; i < K; ++i) {
            for (Integer member: tmpCluster[i])
                clusters[i].add(instances.instance(member));
        }
    }

    @Override
    public int numberOfClusters() throws Exception {
        return K;
    }
    
    public void setNumberOfClusters(int K) throws Exception {
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
    
    protected int cluster(Instance instance) {
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
