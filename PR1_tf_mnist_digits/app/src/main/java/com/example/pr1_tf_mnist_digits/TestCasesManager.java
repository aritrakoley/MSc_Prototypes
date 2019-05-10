package com.example.pr1_tf_mnist_digits;

import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;

import java.util.Arrays;

public class TestCasesManager {

    ProcessingUtilities pu;

    TestCasesManager() {
        pu = new ProcessingUtilities();
    }

    public void test_all() {
        test_transformImage();
//        test_getTransform();
//        test_getCenterOfMass();
//        test_padImage();
//        test_matToFlattenedFloatArray();
//        test_fitImage();SS
//        test_to1D();
//        test_to2D();

//        System.out.println("------------TRIM----------------");
//        System.out.println(String.format("Original: (%d, %d)", img.rows(), img.cols()));
//        System.out.println(String.format("Trimmed: (%d, %d)\n\tr1 = %d, c1 = %d, r2 = %d, c2 = %d", img_roi.rows(), img_roi.cols(), r1, c1, r2, c2));
//        System.out.println("------------TRIM [END]----------------");
    }

    private void test_transformImage() {
        String out = "BEGIN Testing 'float[][] transformImage(float[][] img, int shX, int shY)'...\n";
        float a[][] = {{1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {0, 0, 0, 100, 100},
                {0, 0, 0, 100, 100}};


        float[][] ip = a;
        out += "\tInput:\n\t\tIntMat(2D float array) - ( " + ip.length + "x" + ip[0].length + " )\n";
        out += pu.prettyPrintToString(ip);

        int tr[] = pu.getTransform(ip);
        float op[][] = pu.transformImage(ip, tr[0], tr[1]);

        out += "Output:\n" + pu.prettyPrintToString(op);
        out += "FINISHED Testing 'float[][] transformImage(float[][] img, int shX, int shY)'!\n";

        System.out.println(out);
    }

    private void test_getTransform() {
        String out = "BEGIN Testing 'int[] getTransform(float[][] img)'...\n";
        float a[][] = {{1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {0, 0, 0, 100, 100},
                {0, 0, 0, 100, 100}};

        float[][] ip = a;
        out += "\tInput:\n\t\tIntMat(2D float array) - ( " + ip.length + "x" + ip[0].length + " )\n";
        out += pu.prettyPrintToString(ip);

        int tr[] = pu.getTransform(ip);

        out += "Output:\n" + Arrays.toString(tr) + "\n";
        out += "FINISHED Testing 'int[] getTransform(float[][] img)'!\n";

        //-1, 0
        System.out.println(out);
    }

    private void test_getCenterOfMass() {
        String out = "BEGIN Testing 'float[] getCenterOfMass(float[][] img)'...\n";
        float a[][] = {{1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {0, 0, 0, 100, 100},
                {0, 0, 0, 100, 100}};

        float[][] ip = a;
        out += "\tInput:\n\t\tIntMat(2D float array) - ( " + ip.length + "x" + ip[0].length + " )\n";
        out += pu.prettyPrintToString(ip);

        float CoM[] = pu.getCenterOfMass(ip);

        out += "Output:\n" + Arrays.toString(CoM) + "\n";
        out += "FINISHED Testing 'float[] getCenterOfMass(float[][] img)'!\n";

        //3.4634146341463414, 2.451219512195122
        System.out.println(out);
    }

    private void test_padImage() {

        String out = "BEGIN Testing 'float[][]  padImage(float[][] img, int reqr, int reqc)'...\n";
        float[][] a = {{1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1}};
        int req_r = 7, req_c = 7;

        float[][] ip = a;
        out += "\tInput:\n\t\tIntMat(2D float array) - ( " + ip.length + "x" + ip[0].length + " )\n";
        out += pu.prettyPrintToString(ip);
        out += "req_r: " + req_r + "\n";
        out += "req_c: " + req_c + "\n";

        float[][] op = pu.padImage(ip, req_r, req_c);

        out += "Output:\n" + pu.prettyPrintToString(op);
        out += "FINISHED Testing 'float[][]  padImage(float[][] img, int reqr, int reqc)'!\n";

        System.out.println(out);
    }

    public void test_fitImage() {
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

    public void test_matToFlattenedFloatArray() {
        String out = "BEGIN Testing 'float[] matToFlattenedFloatArray(Mat img_mat)'...\n";
        float[][] ar2D = {{1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12}};
        Mat src_mat = pu.floatArrayToIntMat(ar2D);

        float[][] ip = ar2D;
        out += "\tInput:\n\t\t2D float array - ( " + ip.length + "x" + ip[0].length + " )\n";
        out += pu.prettyPrintToString(ip);

        float[] op = pu.matToFlattenedFloatArray(src_mat);

        out += "Output:\n" + Arrays.toString(op) + "\n";
        out += "float[] matToFlattenedFloatArray(Mat img_mat)'!\n";

        System.out.println(out);
    }

    public void test_to1D() {
        String out = "BEGIN Testing 'float[] to1D(float[][] ar)'...\n";
        float[][] ar2D = {{1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12}};

        float[][] ip = ar2D;
        out += "\tInput:\n\t\t2D float array - ( " + ip.length + "x" + ip[0].length + " )\n";
        out += pu.prettyPrintToString(ip);

        float[] ar1D = pu.to1D(ar2D);

        out += "Output:\n" + Arrays.toString(ar1D) + "\n";
        out += "FINISHED Testing 'float[] to1D(float[][] ar)'!\n";

        System.out.println(out);
    }

    public void test_to2D() {
        String out = "BEGIN Testing 'float[][] to1D(float[] ar, int r, int c)'...\n";
        float[] ar1D = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        int new_r = 3;
        int new_c = 4;

        float[] ip = ar1D;
        out += "\tInput:\n\t\t1D float array - ( " + ip.length + " )\n";
        out += Arrays.toString(ip) + "\n";
        out += "\tNew Dims:\n\t\trows = " + new_r + "\n\t\tcols = " + new_c + "\n";

        float[][] ar2D = pu.to2D(ar1D, new_r, new_c);

        out += "Output:\n" + pu.prettyPrintToString(ar2D);
        out += "FINISHED Testing 'float[][] to2D(float[] ar)'!\n";

        System.out.println(out);
    }
}

