package com.example.pr1_tf_mnist_digits;

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

public class ProcessingUtilities {

    //(1) Image manipulation utilities [START]
    public Mat fitImage(Mat img, int max_dim){
        int r = img.rows();
        int c = img.cols();

        if (r > c){
            float factor = (float)max_dim/(float)r;
            r = max_dim;
            c = Math.round(c*factor);
            resize(img, img, new Size(c, r), 0, 0, INTER_AREA);
        }
        else{
            float factor = (float)max_dim/(float)c;
            c = max_dim;
            r = Math.round(r*factor);
            resize(img, img, new Size(c, r), 0, 0, INTER_AREA);
        }
        return img;
    }

    public float[][]  padImage(float[][] img, int reqr, int reqc){
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

    public float[][] transformImage(float[][] img, int shX, int shY){
        Size src_size = new Size(img[0].length, img.length);
        float[][] tr = {{1, 0, shX}, {0, 1, shY}};

        Mat src = floatArrayToIntMat(img);
        Mat dest = new Mat(src_size);
        Mat transform = floatArrayToIntMat(tr);

        warpAffine(src, dest, transform, src_size);

        float[][] shifted = matTo2DFloatArray(dest);
        return shifted;
    }

    public int[] getTransform(float[][] img) {
        int r = img.length;
        int c = img[0].length;
        float CoM[] = getCenterOfMass(img);
        float cx = CoM[0];
        float cy = CoM[1];

        int shX = (int)(Math.round(c/2 - cx));
        int shY =  (int)(Math.round(r/2 - cy));

        int[] shift = {shX, shY};
        return shift;
    }

    public float[] getCenterOfMass(float[][] img){
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
    //Image manipulation utilities [END]



    //(2) Conversion Utilities [START]
    public Mat floatArrayToIntMat(float img_float[][]) {
        int w = img_float[0].length;
        int h = img_float.length;
        Mat img_mat = new Mat(new Size(w, h), CV_8UC1);
        UByteRawIndexer idx = img_mat.createIndexer();
        long i, j;
        for(i=0; i<h; i++)
            for(j=0; j<w; j++)
                idx.put(i, j, (int)img_float[(int)i][(int)j]);

        return img_mat;
    }

    public float[] matToFlattenedFloatArray(Mat img_mat) {

        Mat floatMat = new Mat();
        img_mat.convertTo(floatMat, CV_32F);
        FloatBuffer floatBuffer = floatMat.createBuffer();
        float[] floatArray = new float[floatBuffer.capacity()];
        floatBuffer.get(floatArray);

        return floatArray;
    }

    public float[][] matTo2DFloatArray(Mat img_mat) {
        int r = img_mat.rows();
        int c = img_mat.cols();

        Mat floatMat = new Mat();
        img_mat.convertTo(floatMat, CV_32F);
        FloatBuffer floatBuffer = floatMat.createBuffer();
        float[] img_float = new float[floatBuffer.capacity()];
        floatBuffer.get(img_float);

        float[][] ar2D = to2D(img_float, r, c);
        return ar2D;
    }
    //Conversion Utilities [END]



    //(3) Other Array Utilities [START]
    public float[][] to2D(float[] src, int r, int c) {
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

    public float[] to1D(float[][] src) {
        int r = src.length;
        int c = src[0].length;

        float[] dest = new float[r*c];

        int i, j, k=0;
        for(i=0; i<r; i++)
            for(j=0; j<c; j++, k++)
                dest[k] = src[i][j];

        return dest;
    }

    public int argmax(float[] a){
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

    public String prettyPrintToString(float[][] ar){
        String out = "[\n";
        for(int i=0; i<ar.length; i++)
            out += (Arrays.toString(ar[i]) + "\n");
        out += "]\n";

        return out;
    }
    //Other Array Utilities [END]
}
