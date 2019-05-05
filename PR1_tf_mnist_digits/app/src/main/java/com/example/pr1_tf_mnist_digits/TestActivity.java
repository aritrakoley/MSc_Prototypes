package com.example.pr1_tf_mnist_digits;

import android.app.Activity;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;

public class TestActivity extends AppCompatActivity {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_test);

        Button btn_main = findViewById(R.id.btn_main);
        btn_main.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent main_activity_intent = new Intent(TestActivity.this, MainActivity.class);
                startActivity(main_activity_intent);
            }
        });

        //Tests
        Button btn_test = findViewById(R.id.btn_test);
        btn_test.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                TestCasesManager test = new TestCasesManager();
                test.test_all();
            }
        });
    }

    private void displayArrayToConsole(float[][] ar, String title)
    {
        String text = "--------------" + title + "-------------\n";
        text += "Shape: (" + ar.length + ", " + ar[0].length + ")\n";
        int i, j;
        for(i=0; i<ar.length; i++)
        {
            for(j=0; j<ar[i].length; j++)
                text += ar[i][j] + "   ";
            text += "\n";
        }
        text += "Shape: (" + ar.length + ", " + ar[0].length + ")\n";
        text += "--------------DATA [][]-------{end}\n";
        System.out.println(text);
    }

    private float[] stringToFloat(String[] ar){
        float arf[] = new float[ar.length];
        for(int i=0; i<ar.length; i++)
            arf[i] = Float.parseFloat(ar[i]);
        return arf;
    }

    private ByteBuffer loadModelFile(Activity activity, String filename) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(filename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return (ByteBuffer) fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private float[][] getPredictions(float[][] out) {
        float y_pred[][] = new float[out.length][1];
        for(int i=0; i<out.length; i++)
            y_pred[i][0] = argmax(out[i]);
        return y_pred;
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

    private float getAccuracy(float[][] y, float[][] y_pred) {
        int i, j, correct=0, total=y.length;
        for(i=0; i<y.length; i++)
            if(y_pred[i][0] == y[i][0])
                correct++;

        System.out.println("Total: " + total);
        System.out.println("Correct: " + correct);
        return (float)correct/(float)total;
    }

    void evaluate(Activity activity) throws IOException {

        //(0) Required Values
        String image_file = "mnist_digits_test_images.csv";
        String label_file = "mnist_digits_test_labels.csv";
        String modelFilename = "tf_mnist_model.tflite";
        float X_test[][], y_test[][], y_pred[][];
        int ROWS=0, DIMS=0, CLASS_LABELS=10, MAX_ROWS=10;

        //(1) Read test_file data into float arrays
        BufferedReader br_image = new BufferedReader(new InputStreamReader(activity.getAssets().open(image_file)));
        BufferedReader br_label = new BufferedReader(new InputStreamReader(activity.getAssets().open(label_file)));

        String image_line, label_line;
        ArrayList image_list = new ArrayList<float[]>();
        ArrayList label_list = new ArrayList<float[]>();
        while((image_line = br_image.readLine()) != null && (label_line = br_label.readLine()) != null){
            ROWS++;
            image_list.add(stringToFloat(image_line.split(",")));
            label_list.add(stringToFloat(label_line.split(",")));
            if (ROWS >= MAX_ROWS)
                break;
        }
        DIMS = ((float[]) image_list.get(0)).length;

        X_test = (float[][]) image_list.toArray(new float[][] {{0}});
        y_test = (float[][]) label_list.toArray(new float[][] {{0}});

        displayArrayToConsole(X_test, "X_test");
        displayArrayToConsole(y_test, "y_test");

        //(2) Make Predictions
        float out[][] = new float[ROWS][CLASS_LABELS];
        Interpreter tflite = new Interpreter(loadModelFile(activity, modelFilename));
        tflite.run(X_test, out);
        tflite.close();
        y_pred = getPredictions(out);

        //(3) Evaluate Predictions
        float acc = getAccuracy(y_test, y_pred);
        System.out.println("Accuracy: " + acc);

        //(4) Display Evaluation in AlertDialog
        AlertDialog.Builder dialogBuilder = new AlertDialog.Builder(activity);
        dialogBuilder.setTitle("Evaluation Results:");
        dialogBuilder.setMessage("Accuracy: " + acc + "\n[ Rows Tested:" + ROWS+  ", Features: " + DIMS + ", Classes: " + CLASS_LABELS +" ]");
        dialogBuilder.setNeutralButton("OK",
                new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        dialog.dismiss();
                    }
                });
        dialogBuilder.create().show();
    }
}
