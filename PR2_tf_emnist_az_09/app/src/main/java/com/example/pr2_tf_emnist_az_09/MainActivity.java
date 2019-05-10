package com.example.pr2_tf_emnist_az_09;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.support.annotation.NonNull;
import android.support.constraint.ConstraintLayout;
import android.support.constraint.ConstraintSet;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.ihhira.android.filechooser.FileChooser;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.tensorflow.lite.Interpreter;
import org.w3c.dom.Text;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_ADAPTIVE_THRESH_MEAN_C;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_THRESH_BINARY_INV;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;
import static org.bytedeco.opencv.global.opencv_imgproc.adaptiveThreshold;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

public class MainActivity extends AppCompatActivity {

    //Global Constants
    final private int PERMISSION_CODE = 1;
    final private int DIM_r = 28, DIM_c = 28;
    final int DIGIT_MODE = 0, LETTER_MODE = 1;
    final int N_LABELS_09 = 10, N_LABELS_AZ = 26;
    final String TFLITE_AZ_FN = "tf_mnist_model_az.tflite";
    final String TFLITE_09_FN = "tf_mnist_model_09.tflite";
    final String STAT_BAR = "| MODE: %d || BLK_SIZE: %d || C: %d || PXL_TH: %d |";
    final int DEFAULT_BLOCK_SIZE = 151; //needs to be an ODD number
    final int DEFAULT_MEAN_C = 20;
    final int DEFAULT_TRIM_PIXEL_THRESHOLD = 1;

    //Global Variables
    int MODE = 0;
    int CLASSES;
    String MODEL_FN;
    int BLOCK_SIZE = 151; //needs to be an ODD number
    int MEAN_C = 20;
    int TRIM_PIXEL_THRESHOLD = 1;

    Mat img;
    ProcessingUtilities pu;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (!checkPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
            requestPermission();
        }

        pu = new ProcessingUtilities();

        updateMode();
        updateStatusBar(MODE, BLOCK_SIZE, MEAN_C, TRIM_PIXEL_THRESHOLD);
    }



    //--------------------------------------------------------------UI Related Code [Start]
    public void onClick(View v) throws IOException {

        if (checkPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
            switch (v.getId()){

                case R.id.iv_in:
                    openFilePicker();
                    break;

                case R.id.btn_proc:
                    process();
                    break;

                case R.id.tv_stat:
                    openSettings();
                    break;

                case R.id.sw_mode:
                    updateMode();
                    break;

                default:
                    Toast.makeText(this, "Unidentified Event Occurred!", Toast.LENGTH_SHORT).show();
            }
        }
        else
            requestPermission();
    }
    //--------------------------------------------------------------UI Related Code [END]


    //--------------------------------------------------------------Settings Related Code [Start]
    private void openSettings() {

        //(1) Make Visible
        makeSettingsVisible();

        //(2) Listen for Button Clicks
        ImageButton btn_save = findViewById(R.id.btn_save);
        ImageButton btn_reset = findViewById(R.id.btn_reset);

        btn_save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String val = null;
                if((val = ((EditText) findViewById(R.id.ed_pxlth)).getText().toString()).length() > 0)
                    TRIM_PIXEL_THRESHOLD = Integer.parseInt(val);

                if((val = ((EditText) findViewById(R.id.ed_blksz)).getText().toString()).length() > 0)
                    BLOCK_SIZE = Integer.parseInt(val);

                if((val = ((EditText) findViewById(R.id.ed_meanc)).getText().toString()).length() > 0)
                    MEAN_C = Integer.parseInt(val);

                updateStatusBar(MODE, BLOCK_SIZE, MEAN_C, TRIM_PIXEL_THRESHOLD);
                makeSettingsGone();
            }
        });

        btn_reset.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                TRIM_PIXEL_THRESHOLD = DEFAULT_TRIM_PIXEL_THRESHOLD;
                BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
                MEAN_C = DEFAULT_MEAN_C;

                updateStatusBar(MODE, BLOCK_SIZE, MEAN_C, TRIM_PIXEL_THRESHOLD);
                makeSettingsGone();
            }
        });


    }
    private void makeSettingsGone() {
        ConstraintLayout constraintLayout = findViewById(R.id.parent_cl);
        LinearLayout ll_settings = findViewById(R.id.ll_settings);

        ConstraintSet constraintSet = new ConstraintSet();

        //TV constraints
        constraintSet.connect(R.id.tv_stat, ConstraintSet.BOTTOM, R.id.iv_in, ConstraintSet.TOP);

        //IM constraints
        constraintSet.connect(R.id.iv_in, ConstraintSet.TOP, R.id.tv_stat, ConstraintSet.BOTTOM);

        ll_settings.setVisibility(View.GONE);
        constraintSet.clone(constraintLayout);
        constraintSet.applyTo(constraintLayout);

    }
    private void makeSettingsVisible() {
        ConstraintLayout constraintLayout = findViewById(R.id.parent_cl);
        LinearLayout ll_settings = findViewById(R.id.ll_settings);

        ConstraintSet constraintSet = new ConstraintSet();
        //LL constraint
        constraintSet.connect(R.id.ll_settings, ConstraintSet.TOP, R.id.tv_stat, ConstraintSet.BOTTOM);
        constraintSet.connect(R.id.ll_settings, ConstraintSet.BOTTOM, R.id.iv_in, ConstraintSet.TOP);

        //IM constraints
        constraintSet.connect(R.id.tv_stat, ConstraintSet.BOTTOM, R.id.ll_settings, ConstraintSet.TOP);

        //IM constraints
        constraintSet.connect(R.id.iv_in, ConstraintSet.TOP, R.id.ll_settings, ConstraintSet.BOTTOM);

        ll_settings.setVisibility(View.VISIBLE);
        constraintSet.clone(constraintLayout);
        constraintSet.applyTo(constraintLayout);

    }

    private void updateMode() {
        Boolean mode = ((Switch) findViewById(R.id.sw_mode)).isChecked();
        if(mode == false)
            MODE = 0;
        else
            MODE = 1;

        CLASSES = (MODE == DIGIT_MODE ? N_LABELS_09 : N_LABELS_AZ);
        MODEL_FN = (MODE == DIGIT_MODE ? TFLITE_09_FN : TFLITE_AZ_FN);
        updateStatusBar(MODE, BLOCK_SIZE, MEAN_C, TRIM_PIXEL_THRESHOLD);
    }
    private void updateStatusBar(int mode, int block_size, int mean_c, int trim_pixel_threshold) {
        ((TextView) findViewById(R.id.tv_stat)).setText(String.format(STAT_BAR, MODE, BLOCK_SIZE, MEAN_C, TRIM_PIXEL_THRESHOLD));
    }
    //--------------------------------------------------------------Settings Related Code [END]


    //--------------------------------------------------------------Image Processing and Prediction [Start]
    private float[][] preprocess(Mat segment) {

        //(1) Fit to 20x20 box with aspect ratio preserved
        int max_dim = 20;
        Mat img_fitted = pu.fitImage(segment, max_dim);

        //(*) Convert Mat to Float[][]
        float[][] img_to_pad = pu.matTo2DFloatArray(img_fitted);

        //(2) Pad image to get 28x28 resolution
        float[][] img_padded = pu.padImage(img_to_pad, DIM_r, DIM_c);

        //(3) Translate/Shift smaller image inside 28x28 image based on Center of Mass
        int tr[] = pu.getTransform(img_padded);
        float[][] img_transformed = pu.transformImage(img_padded, tr[0], tr[1]);

        return img_transformed;
    }
    private void process() throws IOException {
        if(img != null){

            //(1) BGR to GRAYSCALE
            Mat img_gray = new Mat(img.size(), CV_8UC1);
            cvtColor(img, img_gray, COLOR_BGR2GRAY);
            System.out.println("Stage 2: (Grayscale Image): " + img_gray.type());

            //(2) Gaussian Blur
            Mat img_blur = new Mat(img.size(), CV_8UC1);
            GaussianBlur(img_gray, img_blur, new Size(3,3), 0);
            System.out.println("Stage 3: (Blur Image): " + img_blur.type());

            //(3) Make BW using Adaptive Threshold
            Mat img_bw = new Mat(img_blur.size(), CV_8UC1);
            adaptiveThreshold(img_blur, img_bw,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV,11,10);
            System.out.println("Stage 4: (BW Image): " + img_bw.type());

            //(5) Segmentation
            ArrayList<Mat> seg_list = pu.segmentImage(img_bw);
            int n_seg = seg_list.size();
            System.out.println("Stage 5: (Each Segment): " + seg_list.get(0).type());


            //(6) Preprocess Each Segment and Make Predictions
            int predictions[] = new int[n_seg];
            float img_preprocessed[][], out[][] = new float[1][CLASSES];
            float[][] ip_tensor = new float[1][DIM_r * DIM_c];

            Interpreter tflite = new Interpreter(loadModelFile(this, MODEL_FN));
            for(int i=0; i<n_seg; i++) {
                //(1) Preprocess segment to better resemble MNIST images
                img_preprocessed = preprocess(seg_list.get(i));
                seg_list.set(i, pu.floatArrayToIntMat(img_preprocessed));

                //(2) Flatten array to make compatible with tflite input tensor,
                // i.e., MxN float32 array where M = no. of tuples, N = features in each tuple
                //and scaling the features
                ip_tensor[0] = pu.to1D(img_preprocessed);
                for(int x=0; x<ip_tensor[0].length; x++)
                    ip_tensor[0][x] = ip_tensor[0][x] / 255.0f;
                tflite.run(ip_tensor, out);
                predictions[i] = pu.argmax(out[0]);
                System.out.println("OUT: " + Arrays.toString(out[0]));
            }
            tflite.close();

            //(7) Display Segments
//            for(int i=0; i<seg_list.size(); i++) {
//                addImage(pu.matToBitmap(seg_list.get(i)), "Prediction: " + predictions[i], "Segment: " + i);
//            }
            //Show Predictions
            ((TextView) findViewById(R.id.tv_pred))
                    .setText(Arrays.toString(predictions));
        }
    }
    //--------------------------------------------------------------Image Processing and Prediction [END]


    //--------------------------------------------------------------Permissions Specific Code [START]
    private void requestPermission() {
        //If no granting, then  dialog explaining why permissions are required.
        ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSION_CODE);
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if( requestCode == PERMISSION_CODE ) //If its the right set of permissions
        {
            if(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED)
                System.out.println("onRequestPermissionsResult: PERMISSION GRANTED");
            else
                System.out.println("onRequestPermissionsResult: PERMISSION NOT GRANTED");
        }
    }
    private boolean checkPermission(String permission){
        int check = ContextCompat.checkSelfPermission(this, permission);
        return (check == PackageManager.PERMISSION_GRANTED);
    }
    //--------------------------------------------------------------Permissions Specific Code [END]


    //--------------------------------------------------------------Other Utilities [START]
    private void openFilePicker() {
        FileChooser fileChooser = new FileChooser(this, "Pick Image", FileChooser.DialogType.SELECT_FILE, null);
        FileChooser.FileSelectionCallback callback = new FileChooser.FileSelectionCallback() {
            @Override
            public void onSelect(File file) {
                if(file != null)
                {
                    img = imread(file.getAbsolutePath());
                    ((ImageView) findViewById(R.id.iv_in)).setScaleType(ImageView.ScaleType.CENTER_INSIDE);
                    ((ImageView) findViewById(R.id.iv_in)).setImageBitmap(pu.matToBitmap(img));
                    System.out.println("Stage 1: (Original Image): " + img.type());
                }
            }
        };
        fileChooser.show(callback);
    }

    private ByteBuffer loadModelFile(Activity activity, String filename) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(filename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return (ByteBuffer) fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    //--------------------------------------------------------------Other Utilities [END]
}
