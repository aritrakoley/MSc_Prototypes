package com.example.pr1_tf_mnist_digits;

import android.graphics.Bitmap;

public class Item {

    private Bitmap image;
    private String title;
    private String desc;

    public Item(Bitmap image, String title, String desc) {
        this.image = image;
        this.title = title;
        this.desc = desc;
    }

    public Bitmap getImage() {
        return image;
    }

    public void setImage(Bitmap image) {
        this.image = image;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getDesc() {
        return desc;
    }

    public void setDesc(String desc) {
        this.desc = desc;
    }
}
