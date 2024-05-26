import Processing.DataProcess;
import Processing.Smote;
import Model.*;
import Utils.Evaluator;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.rules.OneR;
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

public class Main {
    public static void main(String[] args) throws Exception {
//        DataProcess.save_csv2arff("dataset/wind_dataset.csv", "dataset/wind_dataset.arff");

//        Instances data = DataProcess.read_csv_dataset("dataset/wind_dataset_new.csv");
//        data = DataProcess.removeColumn(data, 0); // remove DATE
//        data = DataProcess.fixMissingValues(data);
//        DataProcess.save_instances2arff(data, "dataset/wind_data.arff");

        Instances data = DataProcess.read_arff_dataset("dataset/wind_data.arff");

        System.out.println(data.firstInstance());
        // Print columns name and index
        for (int i = 0; i < data.numAttributes(); i++)
            System.out.println(i + " " + data.attribute(i).name());


        data.setClassIndex(0);
        data.randomize(new Random(507));
        data = DataProcess.normalize(data);

        test(data);

    }

    public static void test(Instances data) throws Exception {
        OneR oner = new OneR();
        RandomForest randomForest = new RandomForest();
        RandomTree randomTree = new RandomTree();
        NaiveBayes naiveBayes = new NaiveBayes();
        J48 j48 = new J48();
        SMO smo = new SMO();
        MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();
        Logistic logistic = new Logistic();

        ArrayList<Classifier> classifiers = new ArrayList<>();
        // Create a VotingEnsemble model
        VotingEnsemble votingEnsemble = new VotingEnsemble(classifiers, 4);

//        classifiers.add(oner);
        classifiers.add(randomForest);
        classifiers.add(randomTree);
        classifiers.add(naiveBayes);
        classifiers.add(j48);
        classifiers.add(smo);
        classifiers.add(logistic);
        classifiers.add(multilayerPerceptron);


        ArrayList<ModelBase> models = new ArrayList<>();
        for (Classifier classifier : classifiers) {

            ModelBase model = new WekaModel(AbstractClassifier.makeCopy(classifier));
            models.add(model);
        }

        // Create a ModelSplitVote object
        ArrayList<ModelBase> modelSplits = new ArrayList<>();
        ModelSplitVote modelSplitVote = new ModelSplitVote(votingEnsemble);
        modelSplits.add(modelSplitVote);
        for (Classifier classifier : classifiers) {
            ModelBase model = new ModelSplit(AbstractClassifier.makeCopy(classifier));
            modelSplits.add(model);
        }

        models.add(votingEnsemble);

        Evaluator evaluator;
        HashMap<String, Integer> params = new HashMap<>();
        params.put("1", 73);
        params.put("2", 73);
        params.put("3", 73);
        params.put("0", 540);

        for (ModelBase model : models) {
            System.out.println("Evaluate using" + model.modelName());
            evaluator = new Evaluator();
            evaluator.k_folds_validation(model, data, 5);
            Random random = new Random(507);

            Smote smote = new Smote(params, 10, "Euclidean", random);
            evaluator.k_folds_validation(model, data, 10, smote);
            System.out.println("---------------------------------");
        }

        for (ModelBase model : modelSplits) {
            System.out.println("Evaluate using" + model.modelName());
            evaluator = new Evaluator();
            evaluator.k_folds_validation(model, data, 10);
            Random random = new Random(507);
            Smote smote = new Smote(params, 5, "Euclidean", random);
            evaluator.k_folds_validation(model, data, 10, smote);
            System.out.println("---------------------------------");
        }

        for (ModelBase model : models) {
            System.out.println("Evaluate using" + model.modelName());
            evaluator = new Evaluator();
            evaluator.n_times_validation(model, data, 10);

            Smote smote = new Smote(params, 5, "Euclidean", new Random(507));
            evaluator.n_times_validation(model, data, 10, smote);
            System.out.println("---------------------------------");
        }


        for (ModelBase model : modelSplits) {
            System.out.println("Evaluate using" + model.modelName());
            evaluator = new Evaluator();
            evaluator.n_times_validation(model, data, 10);
            Random random = new Random(507);
            Smote smote = new Smote(params, 5, "Euclidean", random);
            evaluator.n_times_validation(model, data, 10, smote);
            System.out.println("---------------------------------");
        }

    }
}
