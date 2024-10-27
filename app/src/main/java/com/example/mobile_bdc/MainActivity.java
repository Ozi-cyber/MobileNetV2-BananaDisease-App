package com.example.mobile_bdc;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.example.mobile_bdc.ml.Mobilevit2;
import com.example.mobile_bdc.ml.Model;
import com.example.mobile_bdc.ml.Model1;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;


public class MainActivity extends AppCompatActivity {

    // Declaring Classes
    Button selectBtn, predictBtn, captureBtn;
    TextView result;
    ImageView imageView;
    Bitmap bitmap;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // permission
        getPermission();

        String[] labels = new String[7];
        int cnt=0;
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(getAssets().open("bd_classes.txt")));
            String line=bufferedReader.readLine();
            while(line!=null) {
                labels[cnt]=line;
                cnt++;
                line=bufferedReader.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        selectBtn = findViewById(R.id.selectBtn);
        predictBtn = findViewById(R.id.predictBtn);
        captureBtn = findViewById(R.id.captureBtn);
        result= findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        // Code for "Importing Image from the Phone" Button
        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 10);
            }
        });

        // Code for "Capturing Image from Camera" Button
        captureBtn.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 12);
            }
        });
        //1
        // Code for the "Prediction Button"
        predictBtn.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onClick(View view) {

                if (bitmap == null) {
                    // Show a message to the user indicating no image is selected
                    Toast.makeText(MainActivity.this, "Please select or capture an image first.", Toast.LENGTH_SHORT).show();
                    return; // Exit the method early to prevent further processing
                }

                try {
                    Model model = Model.newInstance(MainActivity.this);

                    // Create input tensor with fixed size
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

                    // Resize the bitmap to 224x224
                    bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

                    // Convert the bitmap to TensorImage
                    TensorImage tensorImage = TensorImage.fromBitmap(bitmap);

                    // Normalize the pixel values
                    float[] normalizedBuffer = new float[224 * 224 * 3];
                    int[] intValues = new int[224 * 224];

                    // Get the pixel values
                    tensorImage.getBitmap().getPixels(intValues, 0, tensorImage.getWidth(), 0, 0, tensorImage.getWidth(), tensorImage.getHeight());

                    // Normalize pixel values to [0, 1]
                    for (int i = 0; i < intValues.length; i++) {
                        // Extract the ARGB values
                        int pixelValue = intValues[i];

                        // Get RGB components
                        int r = (pixelValue >> 16) & 0xFF;
                        int g = (pixelValue >> 8) & 0xFF;
                        int b = pixelValue & 0xFF;

                        // Normalize to [0, 1]
                        normalizedBuffer[i * 3] = r / 255.0f;  // Red
                        normalizedBuffer[i * 3 + 1] = g / 255.0f;  // Green
                        normalizedBuffer[i * 3 + 2] = b / 255.0f;  // Blue
                    }

                    // Load the normalized buffer into the input feature
                    inputFeature0.loadArray(normalizedBuffer);

                    // Run model inference and get result
                    Model.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    result.setText(labels[getMax(outputFeature0.getFloatArray())] + " ");

                    // Releases model resources if no longer used
                    model.close();
                } catch (IOException e) {
                    // Handle the exception
                    e.printStackTrace();
                    Toast.makeText(MainActivity.this, "Error during prediction: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                }

            }
        });

    }

    int getMax(float[] arr){
        int max=0;
        for(int i=0; i<arr.length; i++) {
            if(arr[i] > arr[max]) max=i;
        }
        return max;
    }

    void getPermission(){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, 11);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode==11) {
            if(grantResults.length>0) {
                if(grantResults[0]!=PackageManager.PERMISSION_GRANTED) {
                    this.getPermission();
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode==10) {
            if (data != null) {
                Uri uri = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    imageView.setImageBitmap(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        else if (requestCode==12) {
            bitmap = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(bitmap);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
