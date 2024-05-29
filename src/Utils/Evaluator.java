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
        long total_predicted_time = 0;
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
                double mse = 0.0;
                double mae = 0.0;

                for (Instance instance : test) {
                    try {
                        start_time = System.currentTimeMillis();
                        double predicted = copy_model.classifyInstance(instance);
                        double actual = instance.classValue();

                        mse += Math.pow(predicted - actual, 2);
                        mae += Math.abs(predicted - actual);

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

                mse_accuracy[fold] = mse/test.numInstances();
                mae_accuracy[fold] = mae/test.numInstances();

            }

            out.println("Total average model train time: " + train_elapsed_time/folds + " ms");

            printInformation(mae_accuracy, folds, "mae");
            printInformation(mse_accuracy, folds, "mse");
        } catch (Exception e) {
            e.printStackTrace();
        }
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