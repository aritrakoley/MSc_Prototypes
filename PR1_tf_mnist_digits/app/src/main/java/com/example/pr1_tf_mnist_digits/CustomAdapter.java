package com.example.pr1_tf_mnist_digits;

import android.support.annotation.NonNull;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.ArrayList;

public class CustomAdapter extends RecyclerView.Adapter<CustomAdapter.CustomViewHolder> {

    ArrayList<Item> dataset;
    public CustomAdapter(ArrayList<Item> dataset) {
        this.dataset = dataset;
    }

    // Provide a reference to the views for each data item
    // Complex data items may need more than one view per item, and
    // you provide access to all the views for a data item in a view holder
    public static class CustomViewHolder extends RecyclerView.ViewHolder {
        // each data item is just a string in this case
        public ImageView iv_img;
        public TextView tv_title;
        public TextView tv_desc;

        public CustomViewHolder(View itemView){
            super(itemView);
            iv_img = itemView.findViewById(R.id.iv_img);
            tv_title = itemView.findViewById(R.id.tv_title);
            tv_desc = itemView.findViewById(R.id.tv_desc);
        }
    }

    @NonNull
    @Override
    public CustomAdapter.CustomViewHolder onCreateViewHolder(@NonNull ViewGroup viewGroup, int i) {

        // Inflate the custom layout
        View itemView = LayoutInflater.from(viewGroup.getContext())
                .inflate(R.layout.list_item, viewGroup, false);

        // Return a new holder instance
        CustomViewHolder vh = new CustomViewHolder(itemView);
        return vh;
    }

    @Override
    public void onBindViewHolder(@NonNull CustomViewHolder customViewHolder, int i) {
        Item item = dataset.get(i);

        // Set item views based on your views and data model
        ImageView iv_img = customViewHolder.iv_img;
        TextView tv_title = customViewHolder.tv_title;
        TextView tv_desc = customViewHolder.tv_desc;

        iv_img.setImageBitmap(item.getImage());
        tv_title.setText(item.getTitle());
        tv_desc.setText(item.getDesc());
    }

    @Override
    public int getItemCount() {
        return dataset.size();
    }
}
