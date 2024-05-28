package org.example.model;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

public class VotingEnsemble implements ModelBase {
    private ArrayList<Classifier> m_Classifiers;
    public int num_classes;

    public void copy(VotingEnsemble votingEnsemble) {
        this.m_Classifiers = votingEnsemble.m_Classifiers;
        this.num_classes = votingEnsemble.num_classes;
    }

    public VotingEnsemble(ArrayList<Classifier> classifiers, int num_classes) {
        m_Classifiers = classifiers;
        this.num_classes = num_classes;
    }

    public VotingEnsemble(int num_classes) {
        m_Classifiers = new ArrayList<>();
        this.num_classes = num_classes;
    }

    public void addClassifier(Classifier classifier) {
        this.m_Classifiers.add(classifier);
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        for (Classifier classifier : m_Classifiers) {
            classifier.buildClassifier(instances);
        }
    }

    public double classifyInstance(Instance instance) throws Exception {
        double[] votes = new double[m_Classifiers.size()];

        for (int i = 0; i < m_Classifiers.size(); i++) {
            Classifier classifier = m_Classifiers.get(i);
            // get the predicted class value of the instance from the current classifier
            double predictedClass = classifier.classifyInstance(instance);
            // add the vote of the current classifier to the array of votes
            votes[i] = predictedClass;
        }
        // count the number of votes for each class value
        int[] classCounts = new int[this.num_classes];
        for (int i = 0; i < m_Classifiers.size(); i++) {
            classCounts[(int) votes[i]]++;
        }
        // find the index of the class value with the most votes
        int majorityClassIndex = 0;
        for (int i = 1; i < classCounts.length; i++) {
            if (classCounts[i] > classCounts[majorityClassIndex]) {
                majorityClassIndex = i;
            }
        }
        return majorityClassIndex;
    }

    @Override
    public VotingEnsemble copy() {
        VotingEnsemble votingEnsemble = new VotingEnsemble(this.num_classes);
        for (Classifier classifier : m_Classifiers) {
            try {
                votingEnsemble.addClassifier(AbstractClassifier.makeCopy(classifier));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return votingEnsemble;
    }

    @Override
    public String modelName() {
        return "VotingEnsemble";
    }
}
