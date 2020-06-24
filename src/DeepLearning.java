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
        int[] layerSize = new int[]{5,8};

        /* Load dataset */
        List<List<Integer>> fonts = Datasets.getDataset(1);

        /* Set up layer list */
        List<Integer> layerList = new ArrayList<>();
        for(Integer i : layerSize){
            layerList.add(i);
        }

        /* Loading layer list to encoder and decoder */
        MultiLayerPerceptron encoder = new MultiLayerPerceptron(layerList);
        Collections.reverse(layerList);
        MultiLayerPerceptron decoder = new MultiLayerPerceptron(layerList);

        /* Training *//*
        for(int i = 0; i < totalEpoch; i++) {
            for(int j = 0; j < fonts.size(); j++){
                double error = net.backPropagate(numbers.get(j), outputs.get(j));
            }
        }

        *//* Test *//*
        int a = 0;
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
