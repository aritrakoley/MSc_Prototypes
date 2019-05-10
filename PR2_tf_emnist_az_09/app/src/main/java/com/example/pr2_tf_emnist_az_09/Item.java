package com.example.pr2_tf_emnist_az_09;

import android.graphics.Bitmap;

public class Item {

    static final int BIG = 0;
    static final int SMALL = 1;

    private Bitmap image;
    private String title;
    private String desc;
    private int type;

    public Item(Bitmap image, String title, String desc, int type) {
        this.image = image;
        this.title = title;
        this.desc = desc;
        this.type = type;
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

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }
}
