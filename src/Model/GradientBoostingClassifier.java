package Model;


import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class GradientBoostingClassifier implements ModelBase {

    private AdaBoostM1[] classifiers;
    private double[] weights;
    private int numIterations;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        RandomForest baseClassifier = new RandomForest();
        classifiers = new AdaBoostM1[numIterations];
        weights = new double[numIterations];
        Instances currentData = new Instances(data);

        // Initialize weights
        for (int i = 0; i < numIterations; i++) {
            weights[i] = 1.0 / numIterations;
        }

        // Build classifiers
        for (int i = 0; i < numIterations; i++) {
            classifiers[i] = new AdaBoostM1();
            classifiers[i].setWeightThreshold(1);
            classifiers[i].setNumIterations(1);
            classifiers[i].setClassifier(baseClassifier);
            classifiers[i].buildClassifier(currentData);

            // Update weights
            double error = 0;
            for (int j = 0; j < currentData.numInstances(); j++) {
                Instance instance = currentData.instance(j);
                double prediction = classifiers[i].classifyInstance(instance);
                if (prediction != instance.classValue()) {
                    error += weights[i];
                }
            }
            double beta = error / (1 - error);
            weights[i] = weights[i] * beta;

            // Normalize weights
            double totalWeight = 0;
            for (int j = 0; j < numIterations; j++) {
                totalWeight += weights[j];
            }
            for (int j = 0; j < numIterations; j++) {
                weights[j] = weights[j] / totalWeight;
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double prediction = 0;
        for (int i = 0; i < numIterations; i++) {
            prediction += weights[i] * classifiers[i].classifyInstance(instance);
        }
        return prediction;
    }

    @Override
    public ModelBase copy() throws Exception {

        return new GradientBoostingClassifier();
    }

    @Override
    public String modelName() {
        return "GradientBoostingClassifier";
    }

    public void setNumIterations(int numIterations) {
        this.numIterations = numIterations;
    }
}
