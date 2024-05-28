package org.example.processing;

import weka.core.Instances;

public interface IPreprocess {
    public abstract Instances apply(Instances data) throws Exception;
}
