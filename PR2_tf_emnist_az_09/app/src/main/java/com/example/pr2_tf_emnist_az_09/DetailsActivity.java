package com.example.pr2_tf_emnist_az_09;

import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.bytedeco.opencv.opencv_core.Mat;

import java.io.IOException;
import java.util.ArrayList;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

public class DetailsActivity extends AppCompatActivity {

    private ArrayList out_list;
    private RecyclerView rv_list;
    private RecyclerView.Adapter mAdapter;
    private RecyclerView.LayoutManager mLayoutManager;

    Mat img;
    ProcessingUtilities pu;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_details);

        String img_path = getIntent().getStringExtra("img_path");
        if(img_path != null && !img_path.equals(""))
        {
            pu = new ProcessingUtilities(this);
            img = imread(img_path);

            //(0) RecyclerView Specific Code
            out_list = new ArrayList<Item>();
            mLayoutManager = new LinearLayoutManager(this);
            mAdapter = new CustomAdapter(out_list);

            rv_list = findViewById(R.id.rv_list);
            rv_list.setHasFixedSize(true); // for performance
            rv_list.setLayoutManager(mLayoutManager); // setting layout manager
            rv_list.setAdapter(mAdapter);
            //(0) RecyclerView Specific Code [END]

            if(img != null){
                try {
                    pu.process(img);
                    out_list.addAll(pu.output_list);
                    mAdapter.notifyDataSetChanged();

                    System.out.println("SIZE: " + out_list.size());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            else
                Toast.makeText(this, "Img is null!", Toast.LENGTH_SHORT).show();
        }
        else
            Toast.makeText(this, "Image not found!", Toast.LENGTH_SHORT).show();

    }

}
