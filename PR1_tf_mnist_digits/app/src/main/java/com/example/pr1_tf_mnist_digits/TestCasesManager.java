package com.example.pr1_tf_mnist_digits;

import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.UByteBufferIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;

import java.nio.FloatBuffer;
import java.util.Arrays;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_AREA;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgproc.warpAffine;

public class TestCasesManager {

    public void test_all()
    {

        float a[][] = {{1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {0, 0, 0, 100, 100},
                {0, 0, 0, 100, 100}};

        System.out.println("Float Array[2D]:");
        for (int i = 0; i < a.length; i++)
            System.out.println(Arrays.toString(a[i]));

        float a1D[] = to1D(a);
        System.out.println("Float Array[1D]:");
        for (int i = 0; i < a.length; i++)
            System.out.println(Arrays.toString(a1D));

        Mat mat = floatArrayToMat(a1D, new Size(a[0].length, a.length));

        System.out.println("FloatArray[1D] to Mat:");
        UByteRawIndexer idx = mat.createIndexer();
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++)
                System.out.print(" " + (float) idx.get(i, j));
            System.out.println();
        }

        float[][] new_arr = matToFloatArray(mat);
        System.out.println("Mat to FloatArray[2D]:");
        for (int i = 0; i < new_arr.length; i++)
            System.out.println(Arrays.toString(new_arr[i]));

        float CoM[] = my_center_of_mass(a);
        System.out.println("CoM: " + Arrays.toString(CoM));

        int sh[] = my_get_shift(a);
        System.out.println("Shift: " + Arrays.toString(sh));

        float shifted[][] = my_shifted(a, sh[0], sh[1]);
        System.out.println("Shifted[2D]:");
        for (int i = 0; i < shifted.length; i++)
            System.out.println(Arrays.toString(shifted[i]));

        float[][] a_new = {{1, 1, 1, 1, 1},
                {2, 2, 2, 2, 2},
                {3, 3, 3, 3, 3},
                {4, 4, 4, 4, 4}};
        int reqr = 7;
        int reqc = 7;
        float a_new_p[][] = my_pad(a_new, reqr, reqc);
        System.out.println("Padded[2D]:");
        for (int i = 0; i < a_new_p.length; i++)
            System.out.println(Arrays.toString(a_new_p[i]));

        float[][] a_int = {{1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1}};
        Mat mat_fit = my_fit(a_int, 3);
        float[][] img_fit = matToFloatArray(mat_fit);
        System.out.println("Fit[1D]:");
        for (int i = 0; i < img_fit.length; i++)
            System.out.println(Arrays.toString(img_fit[i]));

    }

    private Mat my_fit(float[][] img, int max_dim){
        int r = img.length;
        int c = img[0].length;

        Mat img_m = floatArrayToMat(to1D(img), new Size(c, r));
        if (r > c){
            float factor = (float)max_dim/(float)r;
            r = max_dim;
            c = Math.round(c*factor);
            resize(img_m, img_m, new Size(c, r), 0, 0, INTER_AREA);
        }
        else{
            float factor = (float)max_dim/(float)c;
            c = max_dim;
            r = Math.round(r*factor);
            resize(img_m, img_m, new Size(c, r), 0, 0, INTER_AREA);
        }

       return img_m;
    }

    private float[][]  my_pad(float[][] img, int reqr, int reqc){
        int r = img.length;
        int c = img[0].length;
        float[][] img_padded = new float[reqr][reqc];

        int row_t = (int)(Math.ceil((reqr-r)/2.0));
        int col_l = (int)(Math.ceil((reqc-c)/2.0));
        System.out.println("row_t: " + row_t + " col_l: " + col_l);
        int i, j, ir, ic;
        for(i=0,ir=row_t; ir<(row_t + r); ir++, i++)
            for(j=0,ic=col_l; ic<(col_l + c); ic++, j++)
                img_padded[ir][ic] = img[i][j];

        return img_padded;
    }


    private float[][] matToFloatArray(Mat img_bw) {
        Mat floatMat = new Mat();
        img_bw.convertTo(floatMat, CV_32F);
        FloatBuffer floatBuffer = floatMat.createBuffer();
        float[][] floatArray = new  float[1][floatBuffer.capacity()];
        floatArray[0] = new float[floatBuffer.capacity()];
        floatBuffer.get(floatArray[0]);

        return floatArray;
    }

    private Mat floatArrayToMat(float img_float[], Size size)
    {
        Mat imgf = new Mat(size, CV_8UC1);
        UByteRawIndexer idx = imgf.createIndexer();
        long i, j;
        for(i=0; i<size.height(); i++)
            for(j=0; j<size.width(); j++)
                idx.put(i, j, ((int) img_float[(int)(i * size.width() + j)]));

        return imgf;
    }

    private Mat floatArrayToMat(float img_float[], Size size, int type)
    {
        Mat imgf = new Mat(size, type);
        FloatRawIndexer idx = imgf.createIndexer();
        long i, j;
        for(i=0; i<size.height(); i++)
            for(j=0; j<size.width(); j++)
                idx.put(i, j, ((int) img_float[(int)(i * size.width() + j)]));

        return imgf;
    }

    private float[] my_center_of_mass(float[][] img){
        float rsum = 0.0F;
        float csum = 0.0F;
        float total =0.0F;
        for(int ir=0; ir<img.length; ir++)
            for(int ic=0; ic<img[0].length; ic++) {
                rsum += ir * img[ir][ic];
                csum += ic * img[ir][ic];
                total += img[ir][ic];
            }
        float cx = csum / total;
        float cy = rsum / total;

        float CoM[] = {cx, cy};
        return CoM;
    }


    private int[] my_get_shift(float[][] img)
    {
        int r = img.length;
        int c = img[0].length;
        float CoM[] = my_center_of_mass(img);
        float cx = CoM[0];
        float cy = CoM[1];

        int shX = (int)(Math.round(c/2 - cx));
        int shY =  (int)(Math.round(r/2 - cy));

        int[] shift = {shX, shY};
        return shift;
    }

    private float[][] my_shifted(float[][] img, int shX, int shY){
        Size src_size = new Size(img[0].length, img.length);
        Mat src = floatArrayToMat(to1D(img), src_size, CV_32F);
        Mat dest = new Mat(src_size);

        float[][] transform = {{1, 0, shX},
                {0, 1, shY}};
        Mat tr = floatArrayToMat(to1D(transform), new Size(3,2), CV_32F);
        warpAffine(src, dest, tr, src_size);

        float[][] shifted = matToFloatArray(dest);
        return shifted;
    }

    private int argmax(float[] a){
        int i_max = 0;
        float max = a[i_max];
        for(int i=1; i<a.length; i++)
            if(a[i] > max)
            {
                max = a[i];
                i_max = i;
            }
        return i_max;
    }

    private float[][] to2D(float[] src, int r, int c)
    {
        if((r * c) == src.length){
            float dest[][] = new float[r][c];
            int i, j, k=0;
            for(i=0; i<r; i++)
                for(j=0; j<c; j++, k++)
                    dest[i][j] = src[k];

            return dest;
        }
        else
            return null;
    }

    private float[] to1D(float[][] src)
    {
        int r = src.length;
        int c = src[0].length;

        float[] dest = new float[r*c];

        int i, j, k=0;
        for(i=0; i<r; i++)
            for(j=0; j<c; j++, k++)
                dest[k] = src[i][j];

        return dest;
    }



}