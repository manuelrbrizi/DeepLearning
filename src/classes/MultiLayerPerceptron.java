package classes;

import java.util.ArrayList;
import java.util.List;

public class MultiLayerPerceptron {

    public List<Layer> layers;

    public MultiLayerPerceptron(List<Integer> layerDisp){

        layers = new ArrayList<>();
        layers.add(new Layer(layerDisp.get(0),0));

        for(int i = 1; i < layerDisp.size(); i++) {
            layers.add(new Layer(layerDisp.get(i),layerDisp.get(i-1)));
        }
    }


    public List<Double> feedForward(List<Double> input) {

        double excitation;

        List<Double> output = new ArrayList<>();

        // Cargamos el input en la primera capa
        for(int i = 0; i<layers.get(0).size;i++){
            layers.get(0).perceptrons.get(i).activation = input.get(i);
        }


        // Recorremos todas las capas empezando con la primera oculta
        for(int i = 1; i < layers.size(); i++){
            // Recorremos todos los perceptrones de una layer
            for(int j = 0; j < layers.get(i).size; j++){
                excitation = 0.0;
                // Y todos los perceptrones de la layer anterior
                for(int k = 0; k < layers.get(i - 1).size; k++){
                    // Calculamos la exitacion de cada perceptron en la capa i
                    excitation += layers.get(i).perceptrons.get(j).weights.get(k) * layers.get(i - 1).perceptrons.get(k).activation;
                }

                // Sumamos el bias del perceptron
                excitation += layers.get(i).perceptrons.get(j).bias;

                // Calculamos la activacion del perceptron
                layers.get(i).perceptrons.get(j).activation = activationExp(excitation);
            }
        }


        // Tomamos los valores de activacion de la ultima capa que es el output de la red dada la entrada
        for(int i = 0; i <  layers.get(layers.size()-1).size; i++){
            output.add(layers.get(layers.size()-1).perceptrons.get(i).activation);
        }

        return output;
    }

    private double activationExp(double excitation){
        double exp = Math.exp(-excitation);
        return  1/(1+exp);
    }

    private double dExp(double x){
        return (x - Math.pow(x, 2));
    }

    public double backPropagate(List<Double> input, List<Double> output){
        // Propagamos hacia adelante el input y conseguimos el output de la red
        List<Double> predictedOutput = feedForward(input);

        // Calculamos el error entre la salida esperada y la salida obtenida con feed forward
        for(int i = 0; i < layers.get(layers.size()-1).size; i++){
            //Actualizamos el delta de la capa de salida que es el error como lo calculabamos antes
            // (expected - predicted) * g'
            layers.get(layers.size()-1).perceptrons.get(i).delta = (output.get(i) - predictedOutput.get(i)) * dExp(predictedOutput.get(i));
        }
        double error;

        // Empezamos a recorrer desde la ultima capa oculta antes de la capa de output
        for(int i = layers.size() - 2; i >= 0; i--){
            // Recorremos todos sus perceptrones
            for(int j = 0; j < layers.get(i).size; j++){
                error = 0.0;
                // Recorremos todos los perceptrones de la capa siguiente
                for(int k = 0; k < layers.get(i + 1).size; k++){
                    // Acumulamos el error para un perceptron con la capa que le sigue
                    // error += w*delta
                    error += layers.get(i+1).perceptrons.get(k).delta * layers.get(i+1).perceptrons.get(k).weights.get(j);
                }
                // Seteamos el nuevo delta para todos los perceptrones de la capa
                // delta = error*g' = sum(w*delta)*g'
                layers.get(i).perceptrons.get(j).delta = error * dExp(layers.get(i).perceptrons.get(j).activation);
            }

            // Con todos los deltas seteados volvemos a recorrer las layers para actualizar los pesos
            for(int j = 0; j < layers.get(i+1).size; j++){
                for(int k = 0; k < layers.get(i).size; k++){
                    double weight = layers.get(i+1).perceptrons.get(j).weights.get(k);
                    // dw = nu*delta*activation
                    double dw = 0.1 * layers.get(i+1).perceptrons.get(j).delta*layers.get(i).perceptrons.get(k).activation;
                    // weight = w + dw
                    layers.get(i+1).perceptrons.get(j).weights.set(k,weight + dw);
                }
                // Actualizamos los bias de los perceptrones
                // bias = bias + nu*delta
                layers.get(i+1).perceptrons.get(j).bias += 0.1 * layers.get(i+1).perceptrons.get(j).delta;
            }
        }

        double totalError = 0.0;

        for(int i = 0; i < output.size(); i++){
            totalError += Math.abs(predictedOutput.get(i) - output.get(i));
        }

        totalError = totalError / output.size();
        return totalError;
    }



}
