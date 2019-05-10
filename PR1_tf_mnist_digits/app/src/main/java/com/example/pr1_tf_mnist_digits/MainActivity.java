package com.example.pr1_tf_mnist_digits;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.View;
import android.widget.Button;

import com.ihhira.android.filechooser.FileChooser;

import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.UByteBufferIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_ADAPTIVE_THRESH_MEAN_C;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_THRESH_BINARY_INV;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_AREA;
import static org.bytedeco.opencv.global.opencv_imgproc.adaptiveThreshold;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgproc.warpAffine;

public class MainActivity extends AppCompatActivity {

    final int CLASSES = 10;
    final private int PERMISSION_CODE = 1;
    final private int DIM_r = 28, DIM_c = 28;

    private ArrayList out_list;
    private RecyclerView rv_list;
    private RecyclerView.Adapter mAdapter;
    private RecyclerView.LayoutManager mLayoutManager;

    File picked_file;
    Mat img;
    ProcessingUtilities pu;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        pu = new ProcessingUtilities();

        //(0) RecyclerView Specific Code
        out_list = new ArrayList<Item>();
        mLayoutManager = new LinearLayoutManager(this);
        mAdapter = new CustomAdapter(out_list);

        rv_list = findViewById(R.id.rv_list);
        rv_list.setHasFixedSize(true); // for performance
        rv_list.setLayoutManager(mLayoutManager); // setting layout manager
        rv_list.setAdapter(mAdapter);
        //(0) RecyclerView Specific Code [END]


        //Event Listeners
        Button btn_pick = findViewById(R.id.btn_pick);
        Button btn_proc = findViewById(R.id.btn_proc);
        Button btn_test = findViewById(R.id.btn_test);

        if(checkPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {

            btn_pick.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    openFilePicker();

                }
            });

            btn_proc.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    try {
                        process();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });

            btn_test.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    Intent test_activity_intent = new Intent(MainActivity.this, TestActivity.class);
                    startActivity(test_activity_intent);
                }
            });
        }
        else{
            requestPermission();
        }
    }

    private void requestPermission()
    {
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
    private void openFilePicker()
    {
        out_list.clear();
        FileChooser fileChooser = new FileChooser(this, "Pick File", FileChooser.DialogType.SELECT_FILE, null);
        FileChooser.FileSelectionCallback callback = new FileChooser.FileSelectionCallback() {
            @Override
            public void onSelect(File file) {
                if(file != null)
                {
                    picked_file = file;
                    img = imread(picked_file.getAbsolutePath());
                    addImage(pu.matToBitmap(img), "Original Image", "");
                    System.out.println("Stage 1: (Original Image): " + img.type());
                }


            }
        };
        fileChooser.show(callback);
    }

    private void process() throws IOException {
            if(img != null){

                //(1) BGR to BINARY
                int block_size = 151, c = 20;
                Mat img_bw = pu.binarizeImage(img, block_size, c);
                //(*) Show BW Image
                addImage(pu.matToBitmap(img_bw), "Binary Image", "Adaptive Thresholding");

                //(2) Trim BW Image
                img_bw = pu.trim(img_bw, 5);
                //(*) Show BW Image
                addImage(pu.matToBitmap(img_bw), "Trimmed Image", "Trim");

                //(3) Segmentation
                ArrayList<Mat> seg_list = pu.segmentImage(img_bw);
                int n_seg = seg_list.size();


                //(6) Preprocess Each Segment and Make Predictions
                int predictions[] = new int[n_seg];
                float img_preprocessed[][], out[][] = new float[1][CLASSES];
                float[][] ip_tensor = new float[1][DIM_r * DIM_c];

                Interpreter tflite = new Interpreter(loadModelFile(this, "tf_mnist_model_09.tflite"));
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
                for(int i=0; i<seg_list.size(); i++) {
                    addImage(pu.matToBitmap(seg_list.get(i)), "Prediction: " + predictions[i], "Segment: " + i);
                }
            }
    }

    private float[][] preprocess(Mat segment) {

        //(1) Fit to 20x20 box with aspect ratio preserved
        int max_dim = 20;
        Mat img_fitted = pu.fitImage(segment, max_dim);

        float[][] img_to_pad = pu.matTo2DFloatArray(img_fitted);
        //(2) Pad image to get 28x28 resolution
        int reqr = 28, reqc =28;
        float[][] img_padded = pu.padImage(img_to_pad, reqr, reqc);

        //(3) Translate/Shift smaller image inside 28x28 image based on Center of Mass
        int tr[] = pu.getTransform(img_padded);
        float[][] img_transformed = pu.transformImage(img_padded, tr[0], tr[1]);

        return img_transformed;
    }

    private void addImage(Bitmap img, String title, String desc) {
        out_list.add(new Item(img, title, desc));
        mAdapter.notifyDataSetChanged();
    }

    private ByteBuffer loadModelFile(Activity activity, String filename) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(filename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return (ByteBuffer) fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
