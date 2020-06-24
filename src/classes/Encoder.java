package classes;

import java.util.Collections;
import java.util.List;

public class Encoder {
    MultiLayerPerceptron encoder;
    MultiLayerPerceptron decoder;

    public Encoder(List<Integer> layerList){
        this.encoder = new MultiLayerPerceptron(layerList);
        Collections.reverse(layerList);
        this.decoder = new MultiLayerPerceptron(layerList);
    }

    public void train(List<Double> input, List<Double> expected){
        List<Double> firstResult = encoder.feedForward(input);
        List<Double> secondResult = decoder.backPropagate(firstResult, expected);
        encoder.backPropagateOnly(secondResult);
    }
}
