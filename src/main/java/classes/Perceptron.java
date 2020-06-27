package classes;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Perceptron {

    public double activation;
    public List<Double> weights;
    public double		bias;
    public double		delta;

    public Perceptron(int prevLayerSize) {
        Random r = new Random();
        weights = new ArrayList<>();
        bias = r.nextDouble();
        delta = r.nextDouble();
        activation = r.nextDouble();

        for(int i = 0; i < prevLayerSize; i++){
            weights.add(r.nextDouble());
        }
    }

}
