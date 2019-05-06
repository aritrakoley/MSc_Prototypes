package com.example.pr1_tf_mnist_digits;

import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;

import java.util.Arrays;

public class TestCasesManager {

    ProcessingUtilities pu;
    TestCasesManager(){
        pu = new ProcessingUtilities();
    }

    public void test_all(){

        test_matToFlattenedFloatArray();
//        test_fitImage();
//        test_to1D();
//        test_to2D();
    }

    public void test_fitImage(){
        String out = "BEGIN Testing 'Mat fitImage(Mat img, int max_dim)'...\n";
        float[][] a = {{1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1}};

        float[][] b = {{1, 1, 1, 1, 1},
                {2, 2, 2, 2, 2},
                {3, 3, 3, 3, 3},
                {4, 4, 4, 4, 4}};
        int max_dim = 3;

        float[][] ip = b; //change for different test cases
        out += "\tInput:\n\t\tIntMat(2D float array) - ( " + ip.length + "x" + ip[0].length + " )\n";
        out += pu.prettyPrintToString(ip);
        out += "Max dim: " + max_dim + "\n";

        Mat mat_in = pu.floatArrayToIntMat(ip);
        Mat mat_out = pu.fitImage(mat_in, max_dim);

        out += "Output:\n" + pu.prettyPrintToString(pu.matTo2DFloatArray(mat_out));
        out += "FINISHED Testing 'Mat fitImage(Mat img, int max_dim)'!\n";

        System.out.println(out);
    }

    public void test_matToFlattenedFloatArray(){
        String out = "BEGIN Testing 'float[] matToFlattenedFloatArray(Mat img_mat)'...\n";
        float [][] ar2D = {{1,2,3,4},
                           {5,6,7,8},
                           {9,10,11,12}};
        Mat src_mat = pu.floatArrayToIntMat(ar2D);

        float[][] ip = ar2D;
        out += "\tInput:\n\t\t2D float array - ( " + ip.length + "x" + ip[0].length + " )\n";
        out += pu.prettyPrintToString(ip);

        float[] op = pu.matToFlattenedFloatArray(src_mat);

        out += "Output:\n" + Arrays.toString(op) + "\n";
        out += "float[] matToFlattenedFloatArray(Mat img_mat)'!\n";

        System.out.println(out);
    }

    public void test_to1D(){
        String out = "BEGIN Testing 'float[] to1D(float[][] ar)'...\n";
        float [][] ar2D = {{1,2,3,4},
                           {5,6,7,8},
                           {9,10,11,12}};

        float[][] ip = ar2D;
        out += "\tInput:\n\t\t2D float array - ( " + ip.length + "x" + ip[0].length + " )\n";
        out += pu.prettyPrintToString(ip);

        float[] ar1D = pu.to1D(ar2D);

        out += "Output:\n" + Arrays.toString(ar1D) + "\n";
        out += "FINISHED Testing 'float[] to1D(float[][] ar)'!\n";

        System.out.println(out);
    }

    public void test_to2D(){
        String out = "BEGIN Testing 'float[][] to1D(float[] ar, int r, int c)'...\n";
        float [] ar1D = {1,2,3,4,5,6,7,8,9,10,11,12};
        int new_r = 3;
        int new_c = 4;

        float[] ip = ar1D;
        out += "\tInput:\n\t\t1D float array - ( " + ip.length + " )\n";
        out += Arrays.toString(ip) + "\n";
        out += "\tNew Dims:\n\t\trows = " + new_r  + "\n\t\tcols = " + new_c + "\n";

        float[][] ar2D = pu.to2D(ar1D, new_r, new_c);

        out += "Output:\n" + pu.prettyPrintToString(ar2D);
        out += "FINISHED Testing 'float[][] to2D(float[] ar)'!\n";

        System.out.println(out);
    }

//    public void test_all_old() {
//
//
//        float a[][] = {{1, 1, 1, 1, 1},
//                {1, 1, 1, 1, 1},
//                {0, 0, 0, 100, 100},
//                {0, 0, 0, 100, 100}};
//
//        float[][] a_new = {{1, 1, 1, 1, 1},
//                {2, 2, 2, 2, 2},
//                {3, 3, 3, 3, 3},
//                {4, 4, 4, 4, 4}};
//
//        float[][] a_int = {{1, 1, 1, 1, 1},
//                {1, 1, 1, 1, 1},
//                {1, 1, 1, 1, 1},
//                {1, 1, 1, 1, 1}};
//
//        System.out.println("Float Array[2D]:");
//        for (int i = 0; i < a.length; i++)
//            System.out.println(Arrays.toString(a[i]));
//
//        float a1D[] = to1D(a);
//        System.out.println("Float Array[1D]:");
//        for (int i = 0; i < a.length; i++)
//            System.out.println(Arrays.toString(a1D));
//
//        Mat mat = floatArrayToMat(a1D, new Size(a[0].length, a.length));
//
//        System.out.println("FloatArray[1D] to Mat:");
//        UByteRawIndexer idx = mat.createIndexer();
//        for (int i = 0; i < a.length; i++) {
//            for (int j = 0; j < a.length; j++)
//                System.out.print(" " + (float) idx.get(i, j));
//            System.out.println();
//        }
//
//        float[][] new_arr = matToFloatArray(mat);
//        System.out.println("Mat to FloatArray[2D]:");
//        for (int i = 0; i < new_arr.length; i++)
//            System.out.println(Arrays.toString(new_arr[i]));
//
//        float CoM[] = my_center_of_mass(a);
//        System.out.println("CoM: " + Arrays.toString(CoM));
//
//        int sh[] = my_get_shift(a);
//        System.out.println("Shift: " + Arrays.toString(sh));
//
//        float shifted[][] = my_shifted(a, sh[0], sh[1]);
//        System.out.println("Shifted[2D]:");
//        for (int i = 0; i < shifted.length; i++)
//            System.out.println(Arrays.toString(shifted[i]));
//
//
//        int reqr = 7;
//        int reqc = 7;
//        float a_new_p[][] = my_pad(a_new, reqr, reqc);
//        System.out.println("Padded[2D]:");
//        for (int i = 0; i < a_new_p.length; i++)
//            System.out.println(Arrays.toString(a_new_p[i]));
//
//
//        Mat mat_fit = my_fit(a_int, 3);
//        float[][] img_fit = matToFloatArray(mat_fit);
//        System.out.println("Fit[1D]:");
//        for (int i = 0; i < img_fit.length; i++)
//            System.out.println(Arrays.toString(img_fit[i]));
//
//        mat_fit = my_fit(floatArrayToMat(to1D(a_int), new Size(a_int[0].length, a_int.length)), 3);
//        img_fit = matToFloatArray(mat_fit);
//        System.out.println("Fit[1D]-Mat:");
//        for (int i = 0; i < img_fit.length; i++)
//            System.out.println(Arrays.toString(img_fit[i]));
//
//    }
}

