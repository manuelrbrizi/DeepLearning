package classes;

import java.util.ArrayList;
import java.util.List;

public class Encoder {
    MultiLayerPerceptron network;

    public Encoder(List<Integer> layerList){
        this.network = new MultiLayerPerceptron(layerList);
    }

    public void train(List<Double> input, List<Double> expected){
        network.backPropagate(input, expected);
    }

    public List<Double> feedForward(List<Double> input) {
        return network.feedForward(input);
    }

    public void printEncoderInformation(){
        System.out.println("--- AUTOENCODER ---");
        for(int i = 0; i < network.layers.size(); i++) {
            System.out.println("Layer " + i + ": " + network.layers.get(i).size + " neurons\n");
        }
    }

    public List<Double> getLatentValues(List<Double> input, int latentIndex){
        List<Perceptron> perceptrons = network.getLatent(input, latentIndex);
        List<Double> toReturn = new ArrayList<>();

        double sum = 0.0;

        for(int i = 0; i < input.size(); i++){
            sum += input.get(i) * perceptrons.get(0).weights.get(i);
        }

        toReturn.add(sum);

        sum = 0.0;

        for(int i = 0; i < input.size(); i++){
            sum += input.get(i) * perceptrons.get(1).weights.get(i);
        }

        toReturn.add(sum);
        return toReturn;
    }
}
