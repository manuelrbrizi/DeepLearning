package classes;

import java.util.ArrayList;
import java.util.List;

public class Layer {

    public List<Perceptron> perceptrons;
    public int size;


    public Layer(int size, int prevSize)
    {
        this.size = size;
        perceptrons = new ArrayList<>();

        for(int i = 0; i < size; i++){
            perceptrons.add(new Perceptron(prevSize));
        }

    }
}
