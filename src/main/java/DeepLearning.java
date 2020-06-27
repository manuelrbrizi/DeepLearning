import classes.MultiLayerPerceptron;
import utilities.Datasets;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class DeepLearning {

    public static void main(String[] args){

        /* Parameters */
        int totalEpoch = 100000;
        int[] layerSize = new int[]{35,2,35};
        int currentDataset = 3;
        int subListStart = 0;
        int subListEnd = 5;

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
            for (List<Double> font : fonts.subList(subListStart,subListEnd)) {
                encoder.backPropagate(font,font);
            }
        }

        /* Test */
        StringBuilder str = new StringBuilder();
        List<List<Double>> latentPoints = new ArrayList<>();

        for(int font = subListStart; font < subListEnd - 1; font++){
            List<Double> input = fonts.get(font);
            List<Double> output = encoder.feedForward(input);
            List<Double> latentValues = encoder.getLatentValues(input);
            latentPoints.add(latentValues);

            System.out.println();
            System.out.println("--- LATENT ---");
            System.out.printf("X = %f, Y = %f\n", latentValues.get(0), latentValues.get(1));
            str.append(String.format("%1.4f,%1.4f\n", latentValues.get(0), latentValues.get(1)));

            System.out.println();
            System.out.println("--- INPUT ---");
            for(int i = 0; i < input.size(); i++){
                if(i % 5 == 0){
                    System.out.println();
                }
                System.out.printf("%1.0f ", input.get(i));
            }

            System.out.println();
            System.out.println();
            System.out.println("--- OUTPUT ---");

            for(int i = 0; i < output.size(); i++){
                if(i % 5 == 0){
                    System.out.println();
                }
                System.out.printf("%d ", Math.round(output.get(i)));
            }
        }

        /* Load a new pattern to latent space and decode */
        List<Double> point = new ArrayList<>();
        point.add((latentPoints.get(0).get(0) + latentPoints.get(1).get(0) + latentPoints.get(2).get(0))/3);
        point.add((latentPoints.get(0).get(1) + latentPoints.get(1).get(1) + latentPoints.get(2).get(1))/3);
        List<Double> newPattern = encoder.setPointAndFeedForward(point);

        /* Print new pattern */
        System.out.println();
        System.out.println("--- NEW PATTERN ---");
        for(int i = 0; i < newPattern.size(); i++){
            if(i % 5 == 0){
                System.out.println();
            }
            System.out.printf("%1.0f ", newPattern.get(i));
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
