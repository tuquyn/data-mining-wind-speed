package org.example.utils;

import weka.core.Instances;

import java.util.Random;

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