import Processing.DataProcess;
import Processing.Smote;
import Model.*;
import Utils.Evaluator;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.functions.Logistic;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.MultilayerPerceptron;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.lazy.IBk;

import static java.lang.System.out;

public class Main {
    public static void main(String[] args) throws Exception {
//        DataProcess.save_csv2arff("dataset/wind_dataset.csv", "dataset/wind_dataset.arff");

//        Instances data = DataProcess.read_csv_dataset("dataset/wind_dataset_new.csv");
//        data = DataProcess.removeColumn(data, 0); // remove DATE
//        data = DataProcess.fixMissingValues(data);
//        DataProcess.save_instances2arff(data, "dataset/wind_data.arff");

        Instances data = DataProcess.read_arff_dataset("dataset/wind_data.arff");

        out.println(data.firstInstance());
        // Print columns name and index
        for (int i = 0; i < data.numAttributes(); i++)
            out.println(i + " " + data.attribute(i).name());


        data.setClassIndex(0);
        data.randomize(new Random(507));
        data = DataProcess.normalize(data);

        test(data);

    }

    private static ArrayList<Classifier> getClassifiers() {
        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();
        SimpleLinearRegression linearRegression = new SimpleLinearRegression();
        SMOreg smoreg = new SMOreg();
        SimpleLogistic simpleLogistic = new SimpleLogistic();
        ZeroR zeroR = new ZeroR();
        IBk knn = new IBk(5);

        ArrayList<Classifier> classifiers = new ArrayList<>();
        // Create a VotingEnsemble model
        VotingEnsemble votingEnsemble = new VotingEnsemble(classifiers, 4);

//        classifiers.add(multilayerPerceptron);
        classifiers.add(linearRegression);
        classifiers.add(smoreg);
        classifiers.add(zeroR);
        classifiers.add(knn);
        return classifiers;
    }

    public static void test(Instances data) throws Exception {
        out.println(data.firstInstance());
        data.setClassIndex(0);

        ArrayList<Classifier> classifiers = getClassifiers();
//        classifiers.add(simpleLogistic);


        ArrayList<ModelBase> models = new ArrayList<>();
        for (Classifier classifier : classifiers) {

            ModelBase model = new WekaModel(AbstractClassifier.makeCopy(classifier));
            models.add(model);
        }

        // Create a ModelSplitVote object
//        ArrayList<ModelBase> modelSplits = new ArrayList<>();
//        ModelSplitVote modelSplitVote = new ModelSplitVote(votingEnsemble);
//        modelSplits.add(modelSplitVote);
//        for (Classifier classifier : classifiers) {
//            ModelBase model = new ModelSplit(AbstractClassifier.makeCopy(classifier));
//            modelSplits.add(model);
//        }
//
//        models.add(votingEnsemble);

        Evaluator evaluator;
        HashMap<String, Integer> params = new HashMap<>();
        params.put("1", 73);
        params.put("2", 73);
        params.put("3", 73);
        params.put("0", 540);

        out.println("K-folds Validation");
        for (ModelBase model : models) {
            out.println("Evaluate using" + model.modelName());
            evaluator = new Evaluator();
            evaluator.k_folds_validation(model, data, 5);
//            Random random = new Random(507);

//            Smote smote = new Smote(params, 10, "Euclidean", random);
//            evaluator.k_folds_validation(model, data, 10, smote);
            out.println("---------------------------------");
        }

//        for (ModelBase model : modelSplits) {
//            System.out.println("Evaluate using" + model.modelName());
//            evaluator = new Evaluator();
//            evaluator.k_folds_validation(model, data, 10);
//            Random random = new Random(507);
//            Smote smote = new Smote(params, 5, "Euclidean", random);
//            evaluator.k_folds_validation(model, data, 10, smote);
//            System.out.println("---------------------------------");
//        }
        out.println("N-times Validation");
        for (ModelBase model : models) {
            out.println("Evaluate using" + model.modelName());
            evaluator = new Evaluator();
            evaluator.n_times_validation(model, data, 10);

//            Smote smote = new Smote(params, 5, "Euclidean", new Random(507));
//            evaluator.n_times_validation(model, data, 10, smote);
            out.println("---------------------------------");
        }


//        for (ModelBase model : modelSplits) {
//            System.out.println("Evaluate using" + model.modelName());
//            evaluator = new Evaluator();
//            evaluator.n_times_validation(model, data, 10);
//            Random random = new Random(507);
//            Smote smote = new Smote(params, 5, "Euclidean", random);
//            evaluator.n_times_validation(model, data, 10, smote);
//            System.out.println("---------------------------------");
//        }

    }
}
