/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package myclusterer;

import java.util.ArrayList;
import java.util.List;
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
public class MyAgnes extends AbstractClusterer {
    
    public class Cluster {
        final private List<Instance> members;
        public Cluster() {
            members = new ArrayList<>();
        }
        public Cluster(Instance member) {
            members = new ArrayList<>();
            members.add(member);
        }
        public Cluster(List<Instance> members) {
            this.members = new ArrayList<>(members);
        }
        public Cluster(Cluster other) {
            this.members = new ArrayList<>(other.members);
        }
        public Cluster(Cluster cluster1, Cluster cluster2) {
            this.members = new ArrayList<>(cluster1.members);
            members.addAll(cluster2.members);
        }
        public void add(Instance instance) { members.add(instance); }
        public void add(Cluster other) { members.addAll(other.members); }
        
        public int size() { return members.size(); }
        public Instance get(int index) { return members.get(index); }
    }
    
    public enum Linkage {
        SINGLE, COMPLETE
    }
    
    protected Instances instances;
    protected List<Cluster> clusters;
    protected DistanceFunction distanceFunction = new EuclideanDistance();
    protected Linkage linkage = Linkage.SINGLE;
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        runClusterer(new MyAgnes(), args);
    }

    @Override
    public void buildClusterer(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);
        
        this.instances = instances;        
        this.clusters = new ArrayList<>();
        distanceFunction.setInstances(instances);
        
        if (instances.numInstances() == 0) return;
        joinNeighbors();
    }

    @Override
    public int numberOfClusters() throws Exception {
        return clusters.size();
    }    
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        result.enable(Capability.NO_CLASS);

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        result.enable(Capability.STRING_ATTRIBUTES);

        // other
        result.setMinimumNumberInstances(0);
        return result;
    }
    
    public void setLinkage(Linkage linkage) { this.linkage = linkage; }
    public Linkage getLinkage() { return this.linkage; }
    
    private void joinNeighbors() {
        int n = instances.numInstances();
        double[][] distances = new double[n][n];
        for (int i = 0; i < n; ++i) {
            distances[i][i] = 0;
            for (int j = i+1; j < n; ++j) {                                        
                double distance = distanceFunction.distance(
                        instances.instance(i),
                        instances.instance(j));
                distances[i][j] = distances[j][i] = distance;
            }
        }   
        List<Cluster> nodeClusters = new ArrayList<>();        
        for (int i = 0; i < n; ++i) {
            nodeClusters.add(new Cluster(instances.instance(i)));            
            clusters.add(new Cluster(instances.instance(i)));
        }
        while (nodeClusters.size() > 1) {
            double min = Double.MAX_VALUE;
            int firstIdx = -1, secondIdx = -1;
            for (int i = 0; i < nodeClusters.size()-1; ++i) {
                for (int j = i+1; j < nodeClusters.size(); ++j) {
                    double distance = clusterDistance(
                            nodeClusters.get(i),
                            nodeClusters.get(j),
                            linkage);
                    if (distance <= min) {
                        min = distance;
                        firstIdx = i;
                        secondIdx = j;
                    }
                }
            }
            clusters.add(new Cluster(
                nodeClusters.get(firstIdx),
                nodeClusters.get(secondIdx)));
            nodeClusters.add(new Cluster(
                nodeClusters.get(firstIdx),
                nodeClusters.get(secondIdx)));            
            nodeClusters.remove(firstIdx);
            nodeClusters.remove(secondIdx);
        }
    }
    
    private double clusterDistance(Cluster first, Cluster second, Linkage linkage) {
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (int i = 0; i < first.size(); ++i) {
            for (int j = 0; j < second.size(); ++j) {
                double distance = distanceFunction.distance(first.get(i), second.get(j));
                if (distance < min) min = distance;
                if (distance > max) max = distance;
            }
        }
        return linkage == Linkage.SINGLE ? min : max;
    }
       
}
