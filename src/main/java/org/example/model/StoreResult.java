package org.example.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

public class StoreResult {

    @JsonProperty("modelName")
    private String modelName;

    @JsonProperty("mseAccuracy")
    private List<Double> mseAccuracy;

    @JsonProperty("maeAccuracy")
    private List<Double> maeAccuracy;

    public StoreResult(String modelName, List<Double> mseAccuracy, List<Double>maeAccuracy) {
        this.modelName = modelName;
        this.mseAccuracy = mseAccuracy;
        this.maeAccuracy = maeAccuracy;
    }

    public StoreResult() {
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public List<Double> getMseAccuracy() {
        return mseAccuracy;
    }

    public void setMseAccuracy(List<Double> mseAccuracy) {
        this.mseAccuracy = mseAccuracy;
    }

    public List<Double> getMaeAccuracy() {
        return maeAccuracy;
    }

    public void setMaeAccuracy(List<Double> maeAccuracy) {
        this.maeAccuracy = maeAccuracy;
    }
}
