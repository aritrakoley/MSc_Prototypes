package com.example.pr1_tf_mnist_digits;

import android.graphics.Bitmap;

import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_ADAPTIVE_THRESH_MEAN_C;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_THRESH_BINARY_INV;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_AREA;
import static org.bytedeco.opencv.global.opencv_imgproc.adaptiveThreshold;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgproc.warpAffine;

public class ProcessingUtilities {

    //(1) Image manipulation utilities [START]
    Mat binarizeImage(Mat img, int block_size, int c){
        Mat img_gray = new Mat(img.size(), CV_8UC1);
        cvtColor(img, img_gray, COLOR_BGR2GRAY);
        Mat img_bin = new Mat(img_gray.size(), CV_8UC1);
        adaptiveThreshold(img_gray, img_bin, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, block_size, c);

        return img_bin;
    }
    Mat trimImage(Mat img, int pixel_threshold){
        int r1=0,c1=0, r2=(img.rows()-1),c2=(img.cols()-1);
        float th = (255 * pixel_threshold);
        float[][] img_flt = matTo2DFloatArray(img);

        while(sum(img_flt, 0, r1) <= th)
            r1++;

        while(sum(img_flt, 0, r2) <= th)
            r2--;

        while(sum(img_flt, 1, c1) <= th)
            c1++;

        while(sum(img_flt, 1, c2) <= th)
            c2--;

        Mat img_roi = new Mat(img, new Rect(c1,r1, c2-c1, r2-r1));
        return img_roi;
    }
    ArrayList<Mat> segmentImage(Mat img_in){

        int  ic, ir, cnt=0, flag, pxl, m1 = -1, m2 = -1;
        int r = img_in.rows();
        int c = img_in.cols();

        ArrayList seg_list = new ArrayList<Mat>();
        Mat seg;

        UByteRawIndexer idx = img_in.createIndexer();
        for(ic=0; ic<c; ic++)
        {
            if(ic == c-1 && m1 > -1)
                m2 = c-1;
            if(m1 > -1 && m2 > -1)
            {
                if(m1 >= c){m1 = c;}
                if(m2 >= c){m2 = c;}
                if(m1 == m2) { continue; } //0 width segment is ignored

                System.out.println(cnt + ": " + m1 + "--" + m2);

                //(1) Get Segment
                seg = new Mat(img_in, new Rect(m1,0, m2-m1, r-1));
                //(2) Height Crop
                seg = heightCrop(seg);

                if(seg != null) {
                    seg_list.add(seg);
                    cnt++;
                }
                m1 = m2 = -1;
            }

            flag = 0;
            for(ir=0; ir<r; ir++)
            {
                pxl = idx.get(ir, ic);
                if(pxl > 0) {
                    flag = 1;
                    break;
                }
            }

            if(flag == 1 && m1 == -1)
                m1 = ic;
            else if(flag == 0 && m1 > -1)
                m2 = ic;
        }

        return seg_list;
    }

    Mat heightCrop(Mat seg){
        UByteRawIndexer idx = seg.createIndexer();
        int r = seg.rows();
        int c = seg.cols();
        int ic, ir, r1, r2, pxl;
        r1=r2=-1;

        for(ir=0; ir<r && r1 < 0; ir++) {
            for(ic=0; ic<c; ic++)
            {
                pxl = idx.get(ir, ic);
                if(pxl > 0) {
                    r1 = ir;
                    break;
                }
            }
        }

        for(ir=(r-1); ir>=0 && r2 < 0; ir--) {
            for(ic=0; ic<c; ic++)
            {
                pxl = idx.get(ir, ic);
                if(pxl > 0) {
                    r2 = ir;
                    break;
                }
            }
        }

        if(r1 == r2) //ignore 0 height segments
            seg = null;
        else
            seg = new Mat(seg, new Rect(0, r1, c, r2-r1));

        return seg;
    }
    Mat fitImage(Mat img, int max_dim){
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
    float[][]  padImage(float[][] img, int reqr, int reqc){
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
    float[][] transformImage(float[][] img, int shX, int shY){
        Size src_size = new Size(img[0].length, img.length);
        float[][] tr = {{1, 0, shX}, {0, 1, shY}};

        Mat src = floatArrayToFloatMat(img);
        Mat dest = new Mat(src_size);
        Mat transform = floatArrayToFloatMat(tr);

        warpAffine(src, dest, transform, src_size); //needs CV_32F array

        float[][] shifted = matTo2DFloatArray(dest);
        return shifted;
    }
    int[] getTransform(float[][] img) {
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
    float[] getCenterOfMass(float[][] img){
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
    Bitmap matToBitmap(Mat mat){
        AndroidFrameConverter converterToBitmap = new AndroidFrameConverter();
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
        Frame frame = converterToMat.convert(mat);
        Bitmap bmp = converterToBitmap.convert(frame);
        return bmp;
    }

    Mat floatArrayToIntMat(float img_float[][]) {
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

    Mat floatArrayToFloatMat(float[][] img_float) {
        int w = img_float[0].length;
        int h = img_float.length;
        Mat img_mat = new Mat(new Size(w, h), CV_32F);
        FloatRawIndexer idx = img_mat.createIndexer();
        long i, j;
        for(i=0; i<h; i++)
            for(j=0; j<w; j++)
                idx.put(i, j, img_float[(int)i][(int)j]);

        return img_mat;
    }

    float[] matToFlattenedFloatArray(Mat img_mat) {

        Mat floatMat = new Mat();
        img_mat.convertTo(floatMat, CV_32F);
        FloatBuffer floatBuffer = floatMat.createBuffer();
        float[] floatArray = new float[floatBuffer.capacity()];
        floatBuffer.get(floatArray);

        return floatArray;
    }

    float[][] matTo2DFloatArray(Mat img_mat) {
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
    float[][] to2D(float[] src, int r, int c) {
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

    float[] to1D(float[][] src) {
        int r = src.length;
        int c = src[0].length;

        float[] dest = new float[r*c];

        int i, j, k=0;
        for(i=0; i<r; i++)
            for(j=0; j<c; j++, k++)
                dest[k] = src[i][j];

        return dest;
    }

    int argmax(float[] a){
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
    float sum(float[][] ar, int axis, int idx){
        float s = 0.0f;
        int l;
        if(axis == 0)
            for(int i=0; i<ar[0].length;i++)
                s = s + ar[idx][i];
        else if(axis == 1)
            for(int i=0; i<ar.length;i++)
                s = s + ar[i][idx];
        else
            s = Float.MAX_VALUE;

        return s;
    }

    String prettyPrintToString(float[][] ar){
        String out = "[\n";
        for(int i=0; i<ar.length; i++)
            out += (Arrays.toString(ar[i]) + "\n");
        out += "]\n";

        return out;
    }
    //Other Array Utilities [END]







}
