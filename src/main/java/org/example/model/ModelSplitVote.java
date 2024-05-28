package org.example.model;

import weka.core.Instance;
import weka.core.Instances;

public class ModelSplitVote implements ModelBase{
    VotingEnsemble windy_or_not_classifier, wind_speed_classifier;
    Instances windyOrNotData, windSpeedData, data;
    public ModelSplitVote(VotingEnsemble model) {
        this.windy_or_not_classifier = model;
        this.wind_speed_classifier = model.copy();
    }
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        this.data = instances;
        this.windyOrNotData = this.convert_WindyOrNot_Data(instances);
        this.windSpeedData = this.extract_WindSpeed_Data(instances);

        this.windy_or_not_classifier.buildClassifier(this.windyOrNotData);
        this.wind_speed_classifier.buildClassifier(this.windSpeedData);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double isWindy = this.windy_or_not_classifier.classifyInstance(instance);
        if (isWindy == 0) {
            return 0; // Not windy
        }
        return this.wind_speed_classifier.classifyInstance(instance);
    }

    @Override
    public ModelBase copy() {
        return new ModelSplitVote(this.windy_or_not_classifier.copy());
    }

    public String modelName() {
        return "ModelSplit Ensemble";
    }
    public Instances convert_WindyOrNot_Data(Instances data) {
        // Convert the target attribute to binary: 0 for not windy, 1 for windy
        Instances newData = new Instances(data);
        // Assuming the attribute index of "wind" is 0
        for (Instance instance : newData) {
            String windSpeed = instance.stringValue(0);
            double isWindy = (Double.parseDouble(windSpeed) > 0) ? 1 : 0;
            instance.setClassValue(isWindy);
        }
        return  newData;
    }
    public Instances extract_WindSpeed_Data(Instances data) {
        // Extract instances where wind speed is greater than 0
        Instances newData = new Instances(data);
        // remove none windy data
        newData.removeIf(instance -> Double.parseDouble(instance.stringValue(0)) <= 0);
        return newData;
    }
}
