package Utils;


import Processing.DataProcess;
import Processing.Preprocess;
import Processing.IPreprocess;
import Model.ModelBase;
import weka.core.*;
import weka.core.Instances;
import java.util.Random;

import static java.lang.System.out;

public class Evaluator {
    public Evaluator() {
    }

    public void k_folds_validation(ModelBase model, Instances data, int folds) {
        Preprocess preprocess = new Preprocess();
        k_folds_validation(model, data, folds, preprocess);
    }

    public void k_folds_validation(ModelBase model, Instances data, int folds, IPreprocess pre_process) {

        double[] mse_accuracy = new double[folds];
        double[] mae_accuracy = new double[folds];
        long train_elapsed_time = 0;
        try {
            for (int fold = 0; fold < folds; fold++) {
                Instances train = pre_process.apply(data.trainCV(folds, fold));
                Instances test = data.testCV(folds, fold);
                long start_time = System.currentTimeMillis();
                ModelBase copy_model = model.copy();
                copy_model.buildClassifier(train);
                long end_time = System.currentTimeMillis();
                long elapsed_time = end_time - start_time;
                train_elapsed_time += elapsed_time;

                mse_accuracy[fold] = this.MSEvalidation(copy_model, test);
                mae_accuracy[fold] = this.MAEvalidation(copy_model, test);

            }

            out.println("Total average model train time: " + train_elapsed_time/folds + " ms");

            printInformation(mae_accuracy, folds, "mae");
            printInformation(mse_accuracy, folds, "mse");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public double n_times_validation(ModelBase model, Instances data, int n_times) throws Exception {
        Preprocess preprocess = new Preprocess();
        return n_times_validation(model, data, n_times, preprocess);
    }

    public double n_times_validation(ModelBase model, Instances data, int n_times, IPreprocess pre_process) throws Exception {

        double[] mse_accuracy = new double[n_times];
        double[] mae_accuracy = new double[n_times];

        Random random = new Random(507);
        for (int test_time = 0; test_time < n_times; test_time++) {

            TrainTestSplit trainTestSplit = new TrainTestSplit(data, 0.7, random);
            ModelBase copy_model = model.copy();

            Instances trainData = trainTestSplit.train;
            trainData = pre_process.apply(trainData);

            Instances testData = trainTestSplit.test;

            out.println("Analyzing train in evaluate_models " + test_time + " of " + n_times + " tests");
            DataProcess.analyze_data(trainData);
            out.println("Analyzing evaluate_models in evaluate_models " + test_time + " of " + n_times + " tests");
            DataProcess.analyze_data(testData);

            copy_model.buildClassifier(trainData);
            mse_accuracy[test_time] = this.MSEvalidation(copy_model, testData);
            mae_accuracy[test_time] = this.MAEvalidation(copy_model, testData);

        }
        printInformation(mae_accuracy, n_times, "mae");
        return printInformation(mse_accuracy, n_times, "mse");
    }

    public double printInformation(double[] accuracy, double folds, String type) {
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

        out.println("Average "+ type +" loss: " + avg_accuracy);
        return avg_accuracy;
    }

    public double MSEvalidation(ModelBase model, Instances data) {
        double mse = 0.0;
        for (Instance instance : data) {
            try {
                double predicted = model.classifyInstance(instance);
                double actual = instance.classValue();
                mse += Math.pow(predicted - actual, 2);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return mse / data.numInstances();
    }


    public double MAEvalidation(ModelBase model, Instances data) {
        double mae = 0.0;
        for (Instance instance : data) {
            try {
                double predicted = model.classifyInstance(instance);
                double actual = instance.classValue();
                mae += Math.abs(predicted - actual);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return mae / data.numInstances();
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