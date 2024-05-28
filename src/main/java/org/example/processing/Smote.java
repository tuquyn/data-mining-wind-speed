package org.example.processing;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Random;

public class Smote implements IPreprocess {
    private String distance_metric = "Euclidean";
    private int windSpeedColumn = 0; // Index of the wind speed attribute

    public Smote(HashMap<String, Integer> sampling_strategy, int K, String distance_metric, Random rand) {
        this.config(sampling_strategy, K, distance_metric, rand);
    }

    public Random rand;
    public HashMap<String, Integer> sampling_strategy;

    public int K;


    private Instances applySmote(Instances data) {
        Instances balancedData = new Instances(data);

        int k = this.K;

        // Find minority class instances
        HashMap<String, ArrayList<Instance>> classInstances = new HashMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            String classValue = instance.stringValue(windSpeedColumn); // Wind speed attribute
            if (!classInstances.containsKey(classValue)) {
                classInstances.put(classValue, new ArrayList<>());
            }
            classInstances.get(classValue).add(instance);
        }

        for (String classValue : classInstances.keySet()) {
            if (classInstances.get(classValue).size() >= this.sampling_strategy.get(classValue)) continue;
            ArrayList<Instance> currentInstances = classInstances.get(classValue);
            int numSynthetic = this.sampling_strategy.get(classValue) - classInstances.get(classValue).size();
            HashMap<Instance, ArrayList<Instance>> distanceMap = new HashMap<>();

            for (int i = 0; i < numSynthetic; i++) {
                Instance targetInstance = currentInstances.get(rand.nextInt(currentInstances.size()));
                if (!distanceMap.containsKey(targetInstance)) {
                    ArrayList<Instance> kNearestInstances = findKNearestInstance(currentInstances, targetInstance, k);
                    distanceMap.put(targetInstance, kNearestInstances);
                }
                ArrayList<Instance> kNearestInstances = distanceMap.get(targetInstance);
                Instance syntheticInstance = generateSyntheticInstance(targetInstance, kNearestInstances, rand);
                balancedData.add(syntheticInstance);
            }
        }

        return balancedData;
    }

    public Instance generateSyntheticInstance(Instance instance, ArrayList<Instance> neighbors, Random rand) {
        /*
        @params:
            instance: Instance to be synthesized
            neighbor: Instance in the neighborhood of instance
            random: Random number in range [0, 1]
        @return:
            syntheticInstance: Synthetic instance generated from instance and neighbor

        * Generate synthetic instance =   instance + random * (neighbor - instance)
        * If attribute not numeric, synthetic instance's attribute value is the same as instance's
        */
        Instance neighbor = neighbors.get(rand.nextInt(neighbors.size()));
        Instance syntheticInstance = new DenseInstance(instance);
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (instance.attribute(i).isNumeric()) {
                double difference = neighbor.value(i) - instance.value(i);
                syntheticInstance.setValue(i, instance.value(i) + rand.nextDouble() * difference);
            }
        }
        return syntheticInstance;
    }

    private ArrayList<Instance> findKNearestInstance(ArrayList<Instance> data, Instance targetInstance, int k) {
        /*
        @params:
            data: Dataset
            targetInstance: Instance to find k nearest neighbors
            k: Number of nearest neighbors
        @return:
            kNearestInstances: List of k nearest neighbors of targetInstance

         */

        ArrayList<Instance> kNearestInstances = new ArrayList<>();
        for (Instance instance : data) {
            if (instance == targetInstance) {
                continue;
            }
            double distance = calculateDistance(targetInstance, instance);
            DenseInstance temp = new DenseInstance(instance);
            temp.setWeight(distance);
            kNearestInstances.add(temp);
        }
        kNearestInstances.sort(Comparator.comparingDouble(Instance::weight));
        kNearestInstances = new ArrayList<>(kNearestInstances.subList(0, k));

        for (Instance instance : kNearestInstances) {
            instance.setWeight(1);
        }
        return kNearestInstances;
    }

    public int attributeIndex(String name, Instance instance) {
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (instance.attribute(i).name().equals(name)) {
                return i;
            }
        }
        return -1;
    }

    protected double calculateDistance(Instance a, Instance b) {
        /*
        @params:
            a: Instance a
            b: Instance b
        @return:
            distance: Distance between a and b
         */
        if (this.distance_metric.equals("Euclidean")) {
            return EuclideanDistance(a, b);
        }
        if (this.distance_metric.equals("Cosine")) {
            return 1 - CosineSimilarity(a, b);
        }
        return 0;
    }


    public double EuclideanDistance(Instance a, Instance b) {
        double distance = 0;
        for (int i = 0; i < a.numAttributes(); i++) {
            if (a.attribute(i).isNumeric()) {
                distance += Math.pow(a.value(i) - b.value(i), 2);
            }
        }
        return Math.sqrt(distance);
    }

    public double CosineSimilarity(Instance a, Instance b) {
        double dotProduct = 0;
        double normA = 0;
        double normB = 0;
        for (int i = 0; i < a.numAttributes(); i++) {
            if (a.attribute(i).isNumeric()) {
                dotProduct += a.value(i) * b.value(i);
                normA += Math.pow(a.value(i), 2);
                normB += Math.pow(b.value(i), 2);
            }
        }
        return (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
    }

    public void config(HashMap<String, Integer> sampling_strategy, int K, String distance_metric, Random rand) {
        this.sampling_strategy = sampling_strategy;
        this.K = K;
        this.distance_metric = distance_metric;
        this.rand = rand;
    }


    @Override
    public Instances apply(Instances data) {
        return this.applySmote(data);
    }

    public void setDistance(String distance_metric) {
        this.distance_metric = distance_metric;
    }
}
