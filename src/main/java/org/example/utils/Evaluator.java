package org.example.utils;

import org.example.model.ModelBase;
import org.example.model.StoreResult;
import org.example.model.StoreResultList;
import org.example.processing.DataProcess;
import org.example.processing.IPreprocess;
import org.example.processing.Preprocess;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

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
            out.println("Total average model train time: " + train_elapsed_time / folds + " ms");
            printInformation(mae_accuracy, folds, "mae");
            printInformation(mse_accuracy, folds, "mse");

            SaveResult(model.modelName(), mse_accuracy, mae_accuracy, folds);
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
            // out.println(type + " loss, fold " + i + ": " + accuracy[i]);
        }
        avg_accuracy /= folds;

        out.println("Average " + type + " loss: " + avg_accuracy);
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

    public void SaveResult(String modelName, double[] mse_accuracy, double[] mae_accuracy, int folds) {
        List<Double> mseAccuracyList = new ArrayList<Double>();
        List<Double> maeAccuracyList = new ArrayList<Double>();

        for (int fold = 0; fold < folds; fold++) {
            mseAccuracyList.add(mse_accuracy[fold]);
            maeAccuracyList.add(mae_accuracy[fold]);
        }

        StoreResult storeRes = new StoreResult(modelName, mseAccuracyList, maeAccuracyList);
        StoreResultList.getInstance().append(storeRes);
    }
}
