package Utils;


import Processing.Preprocess;
import Processing.IPreprocess;
import Model.ModelBase;
import weka.core.*;
import weka.core.Instances;
import java.util.Random;

public class Evaluator {
    public Evaluator() {
    }

    public void k_folds_validation(ModelBase model, Instances data, int folds) {
        Preprocess preprocess = new Preprocess();
        k_folds_validation(model, data, folds, preprocess);
    }

    public void k_folds_validation(ModelBase model, Instances data, int folds, IPreprocess pre_process) {

        double[] accuracy = new double[folds];

        try {
//            copy_model = new WekaModel(new MultilayerPerceptron());
            for (int fold = 0; fold < folds; fold++) {
                ModelBase copy_model = model.copy();
                Instances train = pre_process.apply(data.trainCV(folds, fold));

                Instances test = data.testCV(folds, fold);
//                System.out.println("Analyzing train in fold " + fold + " of " + folds + " folds");
//                DataProcess.analyze_data(train);
//                System.out.println("Analyzing test in fold " + fold + " of " + folds + " folds");
//                DataProcess.analyze_data(test);
                copy_model.buildClassifier(train);
                accuracy[fold] = this.validation(copy_model, test);
            }
            printInformation(accuracy, folds);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public double n_times_validation(ModelBase model, Instances data, int n_times) throws Exception {
        Preprocess preprocess = new Preprocess();
        return n_times_validation(model, data, n_times, preprocess);
    }

    public double n_times_validation(ModelBase model, Instances data, int n_times, IPreprocess pre_process) throws Exception {

        double[] accuracy = new double[n_times];
        Random random = new Random(507);
        for (int test_time = 0; test_time < n_times; test_time++) {

            TrainTestSplit trainTestSplit = new TrainTestSplit(data, 0.7, random);
            ModelBase copy_model = model.copy();

            Instances trainData = trainTestSplit.train;
            trainData = pre_process.apply(trainData);

            Instances testData = trainTestSplit.test;

//            System.out.println("Analyzing train in test " + test_time + " of " + n_times + " tests");
//            DataProcess.analyze_data(trainData);
//            System.out.println("Analyzing test in test " + test_time + " of " + n_times + " tests");
//            DataProcess.analyze_data(testData);

            copy_model.buildClassifier(trainData);
            accuracy[test_time] = this.validation(copy_model, testData);
        }
        return printInformation(accuracy, n_times);
    }

    public double printInformation(double[] accuracy, double folds) {
        double avg_accuracy = 0, bestAccuracy = 0, worstAccuracy = 1;
        for (int i = 0; i < folds; i++) {
            avg_accuracy += accuracy[i];
            if (accuracy[i] > bestAccuracy) {
                bestAccuracy = accuracy[i];
            }
            if (accuracy[i] < worstAccuracy) {
                worstAccuracy = accuracy[i];
            }

        }
        avg_accuracy /= folds;

        System.out.println("Average accuracy: " + avg_accuracy);
        return avg_accuracy;
//        System.out.println("Best accuracy: " + Math.round(bestAccuracy * 10000.0) / 10000.0);
    }

    public double validation(ModelBase model, Instances data) {
        int match = 0;
        for (Instance instance : data) {
            try {
                int predicted = (int) model.classifyInstance(instance);
                int actual = (int) instance.classValue();
                if (predicted == actual) {
                    match++;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return (double) match / data.numInstances();
    }

}

class TrainTestSplit {
    public Instances train;
    public Instances test;

    public Instances data;
    public double ratio;
    public Random random;

    public TrainTestSplit(Instances data, double ratio, Random random) {
        this.data = new Instances(data);
        this.ratio = ratio;
        this.train = new Instances(data, 0);
        this.test = new Instances(data, 0);
        this.random = random;
        this.data.randomize(this.random);
        int trainSize = (int) Math.round(data.numInstances() * ratio);
        for (int i = 0; i < trainSize; i++) {
            this.train.add(this.data.instance(i));
        }
        for (int i = trainSize; i < data.numInstances(); i++) {
            this.test.add(this.data.instance(i));
        }
    }
}