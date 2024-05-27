import Processing.DataProcess;
import Model.*;
import Utils.Evaluator;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;

import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomTree;

import static java.lang.System.out;

public class Main {
    public static void main(String[] args) throws Exception {

        Instances data = DataProcess.read_arff_dataset("dataset/wind_data.arff");

        out.println("Total number of attributes: " + data.numAttributes());
        Enumeration total_attributes = data.enumerateAttributes();
        while (total_attributes.hasMoreElements()) {
            Attribute attribute = (Attribute) total_attributes.nextElement();
            out.println(attribute);
        }
        out.println("Total number of instances: " + data.numInstances());
        out.println("1st Index\n" + data.firstInstance());
        data.setClassIndex(0);
        data.randomize(new Random(507));
        data = DataProcess.normalize(data);

        test(data);

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

    public static void test(Instances data) throws Exception {
        data.setClassIndex(0);

        ArrayList<Classifier> classifiers = getClassifiers();

        ArrayList<ModelBase> models = new ArrayList<>();
        for (Classifier classifier : classifiers) {

            ModelBase model = new WekaModel(AbstractClassifier.makeCopy(classifier));
            models.add(model);
        }

        Evaluator evaluator;
        int k_num = 10;

        out.println(k_num + "-folds Validation");
        out.println("---------------------------------");

        for (ModelBase model : models) {
            out.println("---------------------------------");
            out.println("Algorithm: " + model.modelName());
            evaluator = new Evaluator();
            evaluator.k_folds_validation(model, data, k_num);
        }
    }
}
