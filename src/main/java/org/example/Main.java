package org.example;

import org.example.model.ModelBase;
import org.example.model.StoreResultList;
import org.example.model.WekaModel;
import org.example.processing.DataProcess;
import org.example.utils.Evaluator;
import org.example.visualize.Visualize;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;

import static java.lang.System.out;

public class Main {
    public static void main(String[] args) throws Exception {
//        Instances data = DataProcess.PreProcess("dataset/wind_dataset_new.csv");
//        DataProcess.SummarizeData(data);
//        evaluate_models(data);

        // write the evaluation result to files
//        StoreResultList.getInstance().writeToFile();

        Visualize.generateCharts();
    }

    private static ArrayList<Classifier> getClassifiers() {

        RandomTree randomTree = new RandomTree();
        RandomForest randomForest = new RandomForest();
        SimpleLinearRegression linearRegression = new SimpleLinearRegression();
        SMOreg smoreg = new SMOreg();
        ZeroR zeroR = new ZeroR();
        IBk knn = new IBk(5);
        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();

        ArrayList<Classifier> classifiers = new ArrayList<>();

        classifiers.add(randomTree);
        classifiers.add(multilayerPerceptron);
        classifiers.add(randomForest);
        classifiers.add(linearRegression);
        classifiers.add(smoreg);
        classifiers.add(zeroR);
        classifiers.add(knn);
        return classifiers;
    }

    public static void evaluate_models(Instances data) throws Exception {
        int k_num = 10;
        Evaluator evaluator = new Evaluator();
        ArrayList<Classifier> classifiers = getClassifiers();
        ArrayList<ModelBase> models = new ArrayList<>();

        for (Classifier classifier : classifiers) {

            ModelBase model = new WekaModel(AbstractClassifier.makeCopy(classifier));
            models.add(model);
        }

        out.println(k_num + "-folds Validation");
        out.println("---------------------------------");

        for (ModelBase model : models) {
            long start_time = System.currentTimeMillis();
            out.println("---------------------------------");
            out.println("Algorithm: " + model.modelName());
            evaluator.k_folds_validation(model, data, k_num);
            long end_time = System.currentTimeMillis();
            long elapsed_time = end_time - start_time;
            out.println("Total run time: " + elapsed_time + " ms");
        }
    }
}
