package com.example.pr2_tf_emnist_az_09;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_ADAPTIVE_THRESH_MEAN_C;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_THRESH_BINARY_INV;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_AREA;
import static org.bytedeco.opencv.global.opencv_imgproc.adaptiveThreshold;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgproc.warpAffine;

public class ProcessingUtilities {

    static final int DEFAULT_BLOCK_SIZE = 151; //needs to be an ODD number
    static final int DEFAULT_MEAN_C = 20;
    static final int DEFAULT_TRIM_PIXEL_THRESHOLD = 3;
    final int DIM_r = 28, DIM_c = 28;
    final int DIGIT_MODE = 0, LETTER_MODE = 1;
    final int N_LABELS_09 = 10, N_LABELS_AZ = 26;
    final String CLASS_LABELS_09 = "0123456789", CLASS_LABELS_AZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    final String[] LABELS = {CLASS_LABELS_09, CLASS_LABELS_AZ};
    final String TFLITE_AZ_FN = "tf_mnist_model_az.tflite";
    final String TFLITE_09_FN = "tf_mnist_model.tflite";
    final String IMG_DESC = "Preprocessed: %s\nResolution: (%d x %d)";


    //Global Variables
    private int MODE;
    private int CLASSES;
    private String MODEL_FN;

    int BLOCK_SIZE; //needs to be an ODD number
    int MEAN_C;
    int TRIM_PIXEL_THRESHOLD;
    Activity activity;
    ArrayList output_list;

    //Constructors
    ProcessingUtilities(Activity act){
        MODE = 0;
        updateMode();

        BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
        MEAN_C = DEFAULT_MEAN_C;
        TRIM_PIXEL_THRESHOLD = DEFAULT_TRIM_PIXEL_THRESHOLD;
        activity = act;
        output_list = new ArrayList<Item>();
    }
    ProcessingUtilities(Activity act, int mode, int pix_th, int block_size, int c){
        MODE = mode;
        updateMode();

        BLOCK_SIZE = block_size;
        MEAN_C = c;
        TRIM_PIXEL_THRESHOLD = pix_th;
        activity = act;
        output_list = new ArrayList<Item>();
    }

    //(0)-------------------------------------------------------------- Main Processing Functions [START]
    int[] process(Mat img) throws IOException {
        int[] predictions = null;
        if(img != null){
            //(0) Add Original Image to output_list
            output_list.add(new Item(matToBitmap(img), "Original Image", "", Item.BIG));

            //(1) BGR to BINARY
            Mat img_bw = binarizeImage(img, BLOCK_SIZE, MEAN_C);
            output_list.add(new Item(matToBitmap(img_bw),
                    "Binary Image",
                    String.format("Adaptive Threshold:\n\tBlock Size: %d\n\tC: %d\n\tResolution: (%d x %d)", BLOCK_SIZE, MEAN_C, img_bw.rows(), img_bw.cols()),
                    Item.BIG));

            //(2) Trim Image
            Mat img_trimmed = trimImage(img_bw, TRIM_PIXEL_THRESHOLD);
            output_list.add(new Item(matToBitmap(img_trimmed),
                    "Trimmed Image",
                    String.format("Trimmed using simple scanning:\n\tPixel Threshold: %d\n\tResolution: (%d x %d)", TRIM_PIXEL_THRESHOLD, img_trimmed.rows(), img_trimmed.cols()),
                    Item.BIG));

            //(3) Segmentation
            ArrayList<Mat> seg_list = segmentImage(img_trimmed);
            int n_seg = seg_list.size();
            System.out.println("Stage 5: (Each Segment): " + seg_list.get(0).type());


            //(4) Preprocess Each Segment and Make Predictions
            predictions = new int[n_seg];
            float[] img_preprocessed; int label;
            Mat o_mat, p_mat;
            for(int i=0; i<n_seg; i++) {
                //(1) Preprocess segment to better resemble MNIST images
                o_mat = seg_list.get(i);
                img_preprocessed = preprocess(o_mat);
                p_mat = preprocessedToMat(img_preprocessed);

                //(2) Predict label for preprocessed image
                label = predict(img_preprocessed);

                //(3) Add label to prediction array
                predictions[i] = label;

                //(4) Add 2 entries to the output_list: segment, pre-processed segment
                output_list.add(new Item(matToBitmap(o_mat),
                        "Segment: " + i,
                        String.format(IMG_DESC, "No", o_mat.rows(), o_mat.cols()),
                        Item.SMALL));
                output_list.add(new Item(matToBitmap(p_mat),
                        "Prediction: " + getLabel(predictions[i]) + " [" + predictions[i] + "]",
                        String.format(IMG_DESC, "Yes", p_mat.rows(), p_mat.cols()),
                        Item.SMALL));
            }

        }
        //(5) Return Predictions
        return predictions;
    }
    float[] preprocess(Mat segment) {

        //(1) Fit to 20x20 box with aspect ratio preserved
        int max_dim = 20;
        Mat img_fitted = fitImage(segment, max_dim);

        //(*) Convert Mat to Float[][]
        float[][] img_to_pad = matTo2DFloatArray(img_fitted);

        //(2) Pad image to get 28x28 resolution
        float[][] img_padded = padImage(img_to_pad, DIM_r, DIM_c);

        //(3) Translate/Shift smaller image inside 28x28 image based on Center of Mass
        int tr[] = getTransform(img_padded);
        float[][] img_transformed = transformImage(img_padded, tr[0], tr[1]);

        //(4) Flatten array to make compatible with tflite input tensor,
        // i.e., MxN float32 array where M = no. of tuples, N = features in each tuple
        //and scaling the features
        float[] img_flattened  = to1D(img_transformed);

        //(5) Feature Scale
        float[] img_scaled = new float[img_flattened.length];
        for(int i=0; i<img_flattened.length; i++)
            img_scaled[i] = img_flattened[i] / 255.0f;

        return img_scaled;
    }
    int predict(float[] img) throws IOException {
        float[][] ip_tensor = new float[1][DIM_r * DIM_c];
        float out[][] = new float[1][CLASSES];

        ip_tensor[0] = img;
        Interpreter tflite = new Interpreter(loadModelFile(activity, MODEL_FN));
        tflite.run(ip_tensor, out);
        tflite.close();

        int pred = argmax(out[0]);
        return pred;
    }

    String getLabel(int pred){
        return String.valueOf(LABELS[MODE].charAt(pred));
    }
    public int getMODE() {
        return MODE;
    }
    public void setMODE(int MODE) {
        this.MODE = MODE;
        updateMode();
    }
    private void updateMode() {
        CLASSES = (MODE == DIGIT_MODE ? N_LABELS_09 : N_LABELS_AZ);
        MODEL_FN = (MODE == DIGIT_MODE ? TFLITE_09_FN : TFLITE_AZ_FN);
    }
    private ByteBuffer loadModelFile(Activity activity, String filename) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(filename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return (ByteBuffer) fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    //(0)-------------------------------------------------------------- Main Processing Functions [END]

    //(1)-------------------------------------------------------------- Image manipulation utilities [START]
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

        Mat fit = new Mat(new Size(20, 20), CV_8UC1);
        System.out.println("FIT: Orig: r="+r+"c="+c);
        if (r > c){
            float factor = (float)max_dim/(float)r;
            r = max_dim;
            c = Math.round(c*factor);
            System.out.println("FIT2: r="+r+"c="+c);
            if(r>0 && c>0)
                resize(img, fit, new Size(c, r), 0, 0, INTER_AREA);
        }
        else{
            float factor = (float)max_dim/(float)c;
            c = max_dim;
            r = Math.round(r*factor);
            System.out.println("FIT2: r="+r+"c="+c);
            if(r>0 && c>0)
                resize(img, fit, new Size(c, r), 0, 0, INTER_AREA);
        }
        return fit;
    }
    float[][]  padImage(float[][] img, int reqr, int reqc){
        int r = img.length;
        int c = img[0].length;
        float[][] img_padded = new float[reqr][reqc];

        int row_t = (int)(Math.ceil((reqr-r)/2.0));
        int col_l = (int)(Math.ceil((reqc-c)/2.0));
        System.out.println("PAD: row_t: " + row_t + " col_l: " + col_l);
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
    //--------------------------------------------------------------Image manipulation utilities [END]


    //(2)-------------------------------------------------------------- Conversion Utilities [START]
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
    private Mat preprocessedToMat(float[] img_preprocessed) {

        //(1) Unscale features
        float[] img_unscaled = new float[img_preprocessed.length];
        for(int i=0; i<img_preprocessed.length; i++)
            img_unscaled[i] = img_preprocessed[i] * 255.0f;

        //(2) Convert to 2D float array then to Mat
        Mat p_mat = floatArrayToIntMat(to2D(img_unscaled, DIM_r, DIM_c));

        return p_mat;
    }
    //--------------------------------------------------------------Conversion Utilities [END]


    //(3)-------------------------------------------------------------- Other Array Utilities [START]
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
    //--------------------------------------------------------------Other Array Utilities [END]







}
