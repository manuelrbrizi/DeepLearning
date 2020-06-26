package classes;

import java.util.Collections;
import java.util.List;

public class Encoder {
    MultiLayerPerceptron encoder;
    MultiLayerPerceptron decoder;
    Layer latent;

    public Encoder(List<Integer> layerList){
        this.encoder = new MultiLayerPerceptron(layerList);
        Collections.reverse(layerList);
        this.decoder = new MultiLayerPerceptron(layerList, 2);
        this.latent = new Layer(2, layerList.get(0));
    }

    public void train(List<Double> input, List<Double> expected){
        List<Double> firstResult = encoder.feedForward(input);
        List<Double> secondResult = decoder.backPropagate(firstResult, expected);
        encoder.backPropagateOnly(secondResult, expected);
    }

    public List<Double> feedForward(List<Double> input) {
        return decoder.feedForward(encoder.feedForward(input));
    }

    public void printEncoderInformation(){
        System.out.println("--- ENCODER ---");
        for(int i = 0; i < encoder.layers.size(); i++){
            System.out.println("Layer " + i + ": " + encoder.layers.get(i).size + " neurons\n");
        }
        System.out.println("--- LATENT ---");
        System.out.println("Layer: " + latent.size + " neurons\n");

        System.out.println("--- DECODER ---");
        for(int i = 0; i < decoder.layers.size(); i++){
            System.out.println("Layer " + i + ": " + decoder.layers.get(i).size + " neurons\n");
        }
    }
}
