package com.example.lanedetect;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import com.example.lanedetect.opencv.LaneDetect;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initLoadOpenCV();
        ActivityCompat.requestPermissions(this, new String[]{
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.CAMERA,
        }, 5);
        DisplayMetrics dm = this.getResources().getDisplayMetrics();
        int screenWidth = dm.widthPixels;
        int screenHeight = dm.heightPixels;
        System.out.println("width:"+screenWidth+", height: "+screenHeight);
        imageView=findViewById(R.id.imageView);
        @SuppressLint("ResourceType") InputStream stream = getResources().openRawResource(R.drawable.test);
        bitmap = BitmapFactory.decodeStream(stream);
        imageView.setImageBitmap(bitmap);
    }
    public void initLoadOpenCV(){
        boolean SUCCESS= OpenCVLoader.initDebug();
        if(SUCCESS){
            Toast.makeText(this.getApplicationContext(),"Loading OpenCV Libraries...",Toast.LENGTH_LONG).show();
        }else{
            Toast.makeText(this.getApplicationContext(),"WARNING: Could not load OpenCV Libraries!",Toast.LENGTH_LONG).show();
        }
    }

    public void detect(View view) {
        LaneDetect ld=new LaneDetect(bitmap,true);
        Mat warp= ld.laneDetection();
        Bitmap newBitMap = Bitmap.createBitmap(warp.width(),warp.height(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(warp,newBitMap);
        imageView.setImageBitmap(newBitMap);
    }
}
