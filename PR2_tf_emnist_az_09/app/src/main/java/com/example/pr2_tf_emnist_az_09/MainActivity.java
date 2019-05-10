package com.example.pr2_tf_emnist_az_09;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
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

    private static final int PERMISSION_CODE = 1;
    final String STAT_BAR = "| MODE: %d || BLK_SIZE: %d || C: %d || PXL_TH: %d |";

    Mat img;
    String img_path;
    ProcessingUtilities pu;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (!checkPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
            requestPermission();
        }

        pu = new ProcessingUtilities(this);
        updateStatusBar();
    }


    //--------------------------------------------------------------UI Related Code [Start]
    public void onClick(View v) throws IOException {

        if (checkPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
            switch (v.getId()){

                case R.id.iv_in:
                    openFilePicker();
                    break;

                case R.id.btn_proc:
                    int[] predictions = pu.process(img);
                    showPredictions(predictions);
                    break;

                case R.id.tv_stat:
                    openSettings();
                    break;

                case R.id.sw_mode:
                    switchMode();
                    break;

                case R.id.btn_details:
                    if(img != null)
                        showDetails();
                    else
                        Toast.makeText(this, "No image selected!", Toast.LENGTH_SHORT).show();
                    break;

                default:
                    Toast.makeText(this, "Unidentified Event Occurred!", Toast.LENGTH_SHORT).show();
            }
        }
        else
            requestPermission();
    }

    private void showDetails() {
        Intent details_intent = new Intent(MainActivity.this, DetailsActivity.class);
        details_intent.putExtra("img_path", img_path);
        startActivity(details_intent);
    }
    private void showPredictions(int[] predictions) {
        String text = "";
        String class_labels[] = {pu.CLASS_LABELS_09, pu.CLASS_LABELS_AZ};
        for(int i=0;i<predictions.length; i++)
            text += (class_labels[pu.getMODE()].charAt(predictions[i]) + " ");
        ((TextView) findViewById(R.id.tv_pred)).setText(text);
    }
    private void updateStatusBar() {
        ((TextView) findViewById(R.id.tv_stat)).setText(String.format(STAT_BAR, pu.getMODE(), pu.BLOCK_SIZE, pu.MEAN_C, pu.TRIM_PIXEL_THRESHOLD));
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
                    pu.TRIM_PIXEL_THRESHOLD = Integer.parseInt(val);

                if((val = ((EditText) findViewById(R.id.ed_blksz)).getText().toString()).length() > 0)
                    pu.BLOCK_SIZE = Integer.parseInt(val);

                if((val = ((EditText) findViewById(R.id.ed_meanc)).getText().toString()).length() > 0)
                    pu.MEAN_C = Integer.parseInt(val);

                updateStatusBar();
                makeSettingsGone();
            }
        });

        btn_reset.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pu.TRIM_PIXEL_THRESHOLD = pu.DEFAULT_TRIM_PIXEL_THRESHOLD;
                pu.BLOCK_SIZE = pu.DEFAULT_BLOCK_SIZE;
                pu.MEAN_C = pu.DEFAULT_MEAN_C;

                updateStatusBar();
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

    private void switchMode() {
        Boolean mode = ((Switch) findViewById(R.id.sw_mode)).isChecked();
        if(mode == false)
            pu.setMODE(0);
        else
            pu.setMODE(1);
        updateStatusBar();
    }
    //--------------------------------------------------------------Settings Related Code [END]


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
                    img_path = file.getAbsolutePath();
                    img = imread(img_path);
                    ((ImageView) findViewById(R.id.iv_in)).setScaleType(ImageView.ScaleType.CENTER_INSIDE);
                    ((ImageView) findViewById(R.id.iv_in)).setImageBitmap(pu.matToBitmap(img));
                    System.out.println("Stage 1: (Original Image): " + img.type());
                }
            }
        };
        fileChooser.show(callback);
    }
    //--------------------------------------------------------------Other Utilities [END]
}
