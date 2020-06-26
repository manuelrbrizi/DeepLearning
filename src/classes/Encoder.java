package classes;

import java.util.Collections;
import java.util.List;

public class Encoder {
    MultiLayerPerceptron encoder;
    MultiLayerPerceptron decoder;
    Layer latent;

    public Encoder(List<Integer> layerList){
        this.latent = new Layer(2, layerList.get(layerList.size()-1));
        this.encoder = new MultiLayerPerceptron(layerList);
        encoder.layers.add(layerList.size(), latent);
        Collections.reverse(layerList);
        this.decoder = new MultiLayerPerceptron(layerList, 2);
        decoder.layers.add(0, latent);
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
        for(int i = 0; i < encoder.layers.size() - 1; i++){
            System.out.println("Layer " + i + ": " + encoder.layers.get(i).size + " neurons\n");
        }
        System.out.println("--- LATENT ---");
        System.out.println("Layer: " + latent.size + " neurons\n");

        System.out.println("--- DECODER ---");
        for(int i = 1; i < decoder.layers.size(); i++){
            System.out.println("Layer " + i + ": " + decoder.layers.get(i).size + " neurons\n");
        }
    }

    public double getLatentX(List<Double> input){
        double sum = 0.0;

        for(int i = 0; i < input.size(); i++){
            sum += input.get(i) * latent.perceptrons.get(0).weights.get(i);
        }

        return sum;
    }

    public double getLatentY(List<Double> input){
        double sum = 0.0;

        for(int i = 0; i < input.size(); i++){
            sum += input.get(i) * latent.perceptrons.get(1).weights.get(i);
        }

        return sum;
    }
}
