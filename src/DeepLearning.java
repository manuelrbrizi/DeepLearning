import classes.Encoder;
import classes.MultiLayerPerceptron;
import utilities.Datasets;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DeepLearning {



    public static void main(String[] args){

        /* Parameters */
        int totalEpoch = 100000;
        int[] layerSize = new int[]{7,14};

        /* Load dataset */
        List<List<Double>> fonts = Datasets.getDataset(1);

        /* Set up layer list */
        List<Integer> layerList = new ArrayList<>();
        for(Integer i : layerSize){
            layerList.add(i);
        }

        /* Loading layer list to encoder */
        Encoder encoder = new Encoder(layerList);

        /* Training */
        for(int i = 0; i < totalEpoch; i++) {
            for (List<Double> font : fonts) {
                encoder.train(font, font);
            }
        }

        /* Test */
        /*int a = 0;
        for(List<Double> n : numbers){
            List<Double> output = net.feedForward(n);
            System.out.printf("input: %d \t",a);
            for (Double out : output){
                System.out.printf("%d\t",Math.round(out));
            }
            a++;
            System.out.println();

        }*/
    }

}
