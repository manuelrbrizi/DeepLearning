import classes.Encoder;
import utilities.Datasets;

import java.util.ArrayList;
import java.util.List;

public class DeepLearning {

    public static void main(String[] args){

        /* Parameters */
        int totalEpoch = 25000;
        int[] layerSize = new int[]{35,14};
        int currentDataset = 3;

        /* Load dataset */
        List<List<Double>> fonts = Datasets.getDataset(currentDataset).subList(0,3);

        /* Set up layer list */
        List<Integer> layerList = new ArrayList<>();
        for(Integer i : layerSize){
            layerList.add(i);
        }

        /* Loading layer list to encoder */
        Encoder encoder = new Encoder(layerList);

        /* Print information */
        encoder.printEncoderInformation();

        /* Training */
        for(int i = 0; i < totalEpoch; i++) {
            for (List<Double> font : fonts) {
                encoder.train(font, font);
            }
        }

        /* Test */
        for(int font = 0; font < 3; font++){
            List<Double> input = fonts.get(font);
            List<Double> output = encoder.feedForward(input);

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
                System.out.printf("%1.0f ", output.get(i));
            }
        }

    }

}
