/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package myclusterer;

import java.util.ArrayList;
import java.util.List;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.NumberOfClustersRequestable;
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
public class MyAgnes extends AbstractClusterer implements NumberOfClustersRequestable {
    
    public class Cluster {
        final private List<Integer> members;        
        
        public Cluster() {
            members = new ArrayList<>();            
        }                
        public Cluster(Integer member) {
            members = new ArrayList<>();
            members.add(member);            
        }        
        public Cluster(Cluster other1, Cluster other2) {
            this.members = new ArrayList<>(other1.members);
            this.members.addAll(other2.members);
        }        
        public void add(Integer member) { members.add(member); }
        public void add(Cluster other) { members.addAll(other.members); }                        
        
        public int size() { return members.size(); }
        public Integer get(int index) { return members.get(index); }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            for (int i = 0; i < members.size(); ++i) {
                if (i > 0) sb.append(", ");
                sb.append(String.valueOf(members.get(i)));
            }
            sb.append("]");
            return sb.toString();
        }
    }       
    
    public enum Linkage {
        SINGLE, COMPLETE
    }
    
    protected Instances instances;
    protected List<Cluster> clusters;
    protected List<List<Cluster>> hierarchy;
    protected DistanceFunction distanceFunction = new EuclideanDistance();
    protected Linkage linkage = Linkage.SINGLE;
    protected int K = 3;
       
    @Override
    public void buildClusterer(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);
        
        this.instances = instances;                
        distanceFunction.setInstances(instances);
        
        if (instances.numInstances() == 0) return;
        joinNeighbors();
    }

    @Override
    public int numberOfClusters() throws Exception {
        return this.K;
    }    
    
    @Override
    public void setNumClusters(int K) throws Exception {
        if (K <= 0) throw new Exception("Number of clusters must be > 0");
        this.K = K;
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
        
        clusters = new ArrayList<>();
        hierarchy = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            Cluster cluster = new Cluster(i);
            clusters.add(cluster);
        }
        hierarchy.add(new ArrayList(clusters));
                
        while (clusters.size() > K) {            
            double min = Double.MAX_VALUE;
            int firstIdx = -1, secondIdx = -1;
            for (int i = 0; i < clusters.size()-1; ++i) {
                for (int j = i+1; j < clusters.size(); ++j) {
                    double distance = clusterDistance(
                            clusters.get(i),
                            clusters.get(j),
                            linkage);
                    if (distance < min) {
                        min = distance;
                        firstIdx = i;
                        secondIdx = j;
                    }
                }
            }            
            Cluster left = clusters.get(firstIdx);
            Cluster right = clusters.get(secondIdx);
            Cluster parent = new Cluster(left, right);            
            clusters.set(firstIdx, parent);
            clusters.remove(secondIdx);
            
            hierarchy.add(new ArrayList(clusters));
        }                   
    }
    
    private double clusterDistance(Cluster first, Cluster second, Linkage linkage) {
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (int i = 0; i < first.size(); ++i) {
            for (int j = 0; j < second.size(); ++j) {
                double distance = distanceFunction.distance(
                        instances.instance(first.get(i)), 
                        instances.instance(second.get(j)));
                if (distance < min) min = distance;
                if (distance > max) max = distance;
            }
        }
        return linkage == Linkage.SINGLE ? min : max;
    }    
    
    @Override
    public int clusterInstance(Instance instance) throws Exception {
        double min = Double.MAX_VALUE;
        int idx = 0;        
        for (int i = 0; i < clusters.size(); ++i) {
            Cluster cluster = clusters.get(i);                        
            for (int j = 0; j < cluster.size(); ++j) {
                double distance = distanceFunction.distance(
                        instance,
                        instances.instance(cluster.get(j)));
                if (distance < min) {
                    min = distance;
                    idx = i;
                }                
            }                       
        }        
        return idx;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();        
        
        sb.append("=== Cluster Hierarchy ===\n");
        for (int i = 0; i < hierarchy.size(); ++i) {
            sb.append("Iteration ").append(String.valueOf(i)).append(":\n");
            for (Cluster cluster: hierarchy.get(i)) sb.append(cluster.toString());
            sb.append("\n\n");
        }
        
        sb.append("=== Cluster Members ===\n");
        for (int i = 0; i < clusters.size(); ++i) {
            Cluster cluster = clusters.get(i);
            if (i > 0) sb.append("\n\n");
            sb.append("Cluster ").append(String.valueOf(i));
            sb.append(": (").append(String.valueOf(cluster.size())).append(" members)\n");
            sb.append(cluster.toString());
        }
        sb.append("\n\n");
        
        return sb.toString();
    }
       
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        runClusterer(new MyAgnes(), args);
    }

}
