import classes.MultiLayerPerceptron;
import utilities.Datasets;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class DeepLearning {

    public static void main(String[] args){

        /* Parameters */
        int totalEpoch = 10000;                // training epoch quantity
        int[] layerSize = new int[]{35,2,35};   // neuron quantity in each layer
        int currentDataset = 3;                 // dataset number (1, 2 or 3)
        int subListStart = 0;                   // first pattern to learn
        int subListEnd = 4;                     // last pattern to learn
        boolean denoisingMode = false;          // false = linear autoencoder, true = denoising autoencoder
        boolean createNewPattern = false;       // decide if autoencoder will create a random pattern
        double noiseProbability = 0.15;         // probability of mutate one bit of a pattern
        int columnWidth = 5;                   // if dataset = 1,2,3 --> width = 5, if dataset = 4 --> width = 10

        /* Load dataset */
        List<List<Double>> fonts = Datasets.getDataset(currentDataset);

        /* Set up layer list */
        List<Integer> layerList = new ArrayList<>();
        for(Integer i : layerSize){
            layerList.add(i);
        }

        /* Loading layer list to encoder */
        MultiLayerPerceptron encoder = new MultiLayerPerceptron(layerList);

        /* Print information */
        encoder.printInformation();

        /* Training */
        for(int i = 0; i < totalEpoch; i++) {
            for (List<Double> font : fonts.subList(subListStart, subListEnd)) {
                if (denoisingMode) {
                    encoder.backPropagate(Datasets.makeNoisy(font, noiseProbability), font);
                } else {
                    encoder.backPropagate(font, font);
                }
            }
        }

        //encoder.printLayerWeights(3);
        System.out.println();
        //encoder.sameWeights();
        encoder.printLayerWeights(1);
        System.out.println();
        //encoder.printLayerWeights(3);

        /* Test */
        StringBuilder str = new StringBuilder();
        List<List<Double>> latentPoints = new ArrayList<>();

        for(int font = subListStart; font < subListEnd; font++){
            List<Double> input = fonts.get(font);
            List<Double> output = encoder.feedForward(input);
            List<Double> latentValues = encoder.getLatentValues(input);
            str.append(String.format("%f,%f\n", latentValues.get(0), latentValues.get(1)));
            latentPoints.add(latentValues);

            System.out.println();
            System.out.println("--- INPUT ---");
            for(int i = 0; i < input.size(); i++){
                if(i % columnWidth == 0){
                    System.out.println();
                }
                System.out.printf("%1.0f ", Math.abs(input.get(i)));
            }

            System.out.println();
            System.out.println();
            System.out.println("--- OUTPUT ---");

            for(int i = 0; i < output.size(); i++){
                if(i % columnWidth == 0){
                    System.out.println();
                }
                System.out.printf("%1.0f ", Math.abs(output.get(i)));
            }
        }

        /* Load a new pattern to latent space and decode */
        if(createNewPattern){
            List<Double> point = new ArrayList<>();
            point.add((latentPoints.get(0).get(0) + latentPoints.get(1).get(0) + latentPoints.get(2).get(0))/3);
            point.add((latentPoints.get(0).get(1) + latentPoints.get(1).get(1) + latentPoints.get(2).get(1))/3);
            List<Double> newPattern = encoder.setPointAndFeedForward(point);

            /* Print new pattern */
            System.out.println();
            System.out.println("--- NEW PATTERN ---");
            for(int i = 0; i < newPattern.size(); i++){
                if(i % columnWidth == 0){
                    System.out.println();
                }
                System.out.printf("%1.0f ", Math.abs(newPattern.get(i)));
            }
        }

        /* Just to print latent space points */
        PrintWriter writer = null;

        try {
            writer = new PrintWriter("latentSpace");
            BufferedWriter out = new BufferedWriter(new FileWriter("latentSpace", true));
            out.write(str.toString());
            out.close();
        }

        catch (IOException e) {
            e.printStackTrace();
        }
    }

}
