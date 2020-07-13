package com.detector.facedetection;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.ToggleButton;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class GenderRecognizer extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "GenderRecognizer";
    private CameraBridgeViewBase mOpenCvCameraView;
    private Net mGenderNet;
    private Net mAgeNet;
    private Net mMaskNet;
    private CascadeClassifier mFaceDetector;
    private File mCascadeFile;
    private Mat mRgba, mGray;
    private int mAbsoluteFaceSize = 0;
    private static final String[] GENDERS = new String[]{"MALE", "FEMALE"};
    private static final String[] AGES = new String[]{"0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60+"};
    private static final String[] MASKS = new String[]{"NO MASK", "MASK"};
    private int mCameraId = 0;
    int camera_width,camera_height;
    private TextView txt_age,txt_gender,txt_mask;
    private LinearLayout lay_nopassing,lay_passing;

    private Mat sobell;
    public static final int CV_32F=5;
    //Connection between app and OpenCV Manager
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    new AsyncTask<Void, Void, Void>() {
                        @Override
                        protected Void doInBackground(Void... voids) {
                            try {
                                //Loading detection classifier from resources
                                InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                                File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                                mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                                FileOutputStream os = new FileOutputStream(mCascadeFile);

                                byte[] buffer = new byte[4096];
                                int bytesRead;
                                while ((bytesRead = is.read(buffer)) != -1) {
                                    os.write(buffer, 0, bytesRead);
                                }
                                is.close();
                                os.close();

                                mFaceDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                                if (mFaceDetector.empty()) {
                                    Log.e(TAG, "Failed to load cascade classfier");
                                    mFaceDetector = null;
                                }else {
                                    Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                                }
                                cascadeDir.delete();

                            }catch (IOException e) {
                                e.printStackTrace();
                                Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                            }
                            return null;
                        }
                    }.execute();

                    mOpenCvCameraView.enableView();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };



    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        }else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.recognizer_gender);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        lay_nopassing = findViewById(R.id.lay_nopassing);
        lay_passing = findViewById(R.id.lay_passing);
        txt_age = findViewById(R.id.txt_age);
        txt_gender = findViewById(R.id.txt_gender);
        txt_mask = findViewById(R.id.txt_mask);

        PulsatorLayout pulsator = (PulsatorLayout) findViewById(R.id.pulsator);
        pulsator.start();

        lay_nopassing.setVisibility(View.GONE);
        lay_nopassing.setVisibility(View.VISIBLE);
        mOpenCvCameraView = findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCameraIndex(mCameraId);
        mOpenCvCameraView.setCvCameraViewListener(this);

        ToggleButton mFlipCamera = findViewById(R.id.toggle_camera);
        mFlipCamera.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    mCameraId = 1;
                    mOpenCvCameraView.disableView();
                    mOpenCvCameraView.setCameraIndex(mCameraId);
                    mOpenCvCameraView.enableView();
                } else {
                    mCameraId = 0;
                    mOpenCvCameraView.disableView();
                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableView();
                }
            }
        });

        Button mBackButton = findViewById(R.id.back_button);
        mBackButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                Methods.reset();
//                Intent backIntent = new Intent(GenderRecognizer.this, WelcomeScreen.class);
//                startActivity(backIntent);
                onBackPressed();
            }
        });
    }

    @SuppressLint("LongLogTag")
    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.d("Debug camera width: ",String.valueOf(width)+" , "+"height: "+String.valueOf(height));
        camera_width=width;
        camera_height=height;

        mGray = new Mat();
        mRgba = new Mat();

        //Loading Caffe model to Dnn
        String proto = getPath("deploy_gender.prototxt", this);
        String weights = getPath("gender_net.caffemodel", this);
        mGenderNet = Dnn.readNetFromCaffe(proto, weights);

        //Loading Caffe model to Dnn
        String protomask = getPath("deploy.prototxt", this);
        String weightsmask = getPath("res10_300x300_ssd_iter_140000.caffemodel", this);
        mMaskNet = Dnn.readNetFromCaffe(protomask, weightsmask);

        String ageproto = getPath("deploy_age.prototxt", this);
        String ageweights = getPath("age_net.caffemodel", this);
        mAgeNet = Dnn.readNetFromCaffe(ageproto, ageweights);

        if (mAgeNet.empty()) {
            Log.i(TAG, "Age: Network loading failed");
        } else {
            Log.i(TAG, "Age: Network loading success");
        }

        if (mGenderNet.empty()) {
            Log.i(TAG, "Gender: Network loading failed");
        }else {
            Log.i(TAG, "Gender: Network loading success");
        }

        if (mMaskNet.empty()) {
            Log.i(TAG, "Mask: Network loading failed");
        }else {
            Log.i(TAG, "Mask: Network loading success");
        }

    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        Point point1 =new Point();
        point1.x=camera_width/2;
        point1.y=camera_height/3;

        Point point2 =new Point();
        point2.x=360;
        point2.y=370;

        Imgproc.circle(mRgba, point1, 400, new Scalar(255, 0, 0, 0), 8);
       // Imgproc.rectangle(mRgba, new Point(camera_width/5,camera_height/5), new Point(camera_width-(camera_width/5),(camera_height/5)*3), new Scalar(63, 81, 181,0),3);

        //        double height_15=camera_height*0.15;
//        double height_minus_15=camera_height-height_15;
//        Point p3 =new Point();
//        p3.x=0;
//        p3.y=height_minus_15;
//        Point p4 =new Point();
//        p4.x=camera_width;
//        p4.y=camera_height;
//
//        Mat Overlay = mRgba.clone();
//        Imgproc.rectangle(mRgba, p3, p4, new Scalar(63, 81, 181,0),CV_FILLED);
//        Core.addWeighted(Overlay,0.25,mRgba,0.4, 0.0, mRgba);
//
//        Mat Overlay_border = mRgba.clone();
//        Imgproc.rectangle(mRgba, p3, p4, new Scalar(63, 81, 181,0),7);
//        Core.addWeighted(Overlay_border,0.25,mRgba,0.75, 0.0, mRgba);
//
//        double posy= ((camera_height - height_minus_15) / 3) + height_minus_15;
//
//        Mat Overlaygender = mRgba.clone();
//        Imgproc.rectangle(mRgba, new Point(50, posy), new Point(camera_width*0.2, posy+((camera_height - height_minus_15) / 3)), new Scalar(255, 255, 255,0),7);
//        Core.addWeighted(Overlaygender,0.25,mRgba,0.75, 0.0, mRgba);



        //Computing absolute face size
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            float mRelativeFaceSize = 0.2f;
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();

        //Using detection classifier
        if (mFaceDetector != null) {
            mFaceDetector.detectMultiScale(mGray, faces, 1.1, 5, 2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }else {
            Log.e(TAG, "Detection is not selected!");
        }
       // lay_passing.setVisibility(View.GONE);
        //Drawing rectangle around detected face
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
             Imgproc.circle(mRgba, point1, 400, new Scalar(0,255, 0, 255), 8);
           // lay_passing.setVisibility(View.VISIBLE);
        }

        //If one face is detected, method predictGender is executed
        if (facesArray.length == 1) {
            final String[] gender_age = predictAgeandGender(mRgba, facesArray).split(",");
          //  String mask = predictMask(mRgba, facesArray);
            //The result of gender recognition
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    lay_nopassing.setVisibility(View.GONE);
                    lay_passing.setVisibility(View.VISIBLE);
                    txt_gender.setText("Gender: "+gender_age[0]);
                    txt_age.setText("Age: "+gender_age[1]);
                }
            });
            try {
                for (Rect face : facesArray) {
                    int posX = (int) Math.max(face.tl().x - 10, 0);
                    int posY = (int) Math.max(face.tl().y - 10, 0);
//                    Imgproc.putText(mRgba, gender_age[0], new Point(50, posy), Core.FONT_HERSHEY_TRIPLEX,
//                            1.5, new Scalar(0, 255, 0, 255));
//                    Imgproc.putText(mRgba, gender_age[1], new Point(camera_width/2, posy), Core.FONT_HERSHEY_TRIPLEX,
//                            1.5, new Scalar(0, 255, 0, 255));
//                    Imgproc.putText(mRgba, gender_age, new Point(posX, posY), Core.FONT_HERSHEY_TRIPLEX,
//                            1.5, new Scalar(0, 255, 0, 255));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        else {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    lay_nopassing.setVisibility(View.VISIBLE);
                    lay_passing.setVisibility(View.GONE);
                    txt_gender.setText("Gender: ");
                    txt_age.setText("Age: ");
                }
            });
        }
        return mRgba;
    }
    public boolean validateFace(int x1, int y1, int x2,
                                int y2, int x, int y)
    {
        if (x > x1 && x < x2 && y > y1 && y < y2)
            return true;
        else
            return false;
    }
    //Method for gender recognition
    private String predictAgeandGender (Mat mRgba, Rect[] facesArray) {
        try {
            for (Rect face : facesArray) {
                Mat capturedFace = new Mat(mRgba, face);
                //Resizing pictures to resolution of Caffe model

                Imgproc.resize(capturedFace, capturedFace, new Size(227, 227));
                //Converting RGBA to BGR
                Imgproc.cvtColor(capturedFace, capturedFace, Imgproc.COLOR_RGBA2BGR);

                //Forwarding picture through Dnn
                Mat inputBlob = Dnn.blobFromImage(capturedFace, 1.0f, new Size(227, 227),
                        new Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);

                mGenderNet.setInput(inputBlob, "data");
                mAgeNet.setInput(inputBlob, "data");
                Mat probsgender = mGenderNet.forward("prob").reshape(1, 1);
                Core.MinMaxLocResult core_gender = Core.minMaxLoc(probsgender); //Getting largest softmax output
                Mat probsage = mAgeNet.forward("prob").reshape(1, 1);
                Core.MinMaxLocResult core_age = Core.minMaxLoc(probsage); //Getting largest softmax output

                double result_gender = core_gender.maxLoc.x; //Result of gender recognition prediction. 1 = FEMALE, 0 = MALE
                double result_age = core_age.maxLoc.x; //Result of age recognition prediction
                final String predictedGender = GENDERS[(int) result_gender];
                final String predictedAge = AGES[(int) result_age];
                Log.i(TAG, "Result is: Gender- " + result_gender+", Age- "+result_age);

                return "Gender: "+predictedGender+", Age: "+predictedAge;
            }
        } catch (Exception e) {
            Log.e(TAG, "Error processing gender", e);
        }
        return null;
    }
    private String predictAge(Mat mRgba, Rect[] facesArray) {
        try {
            for (Rect face : facesArray) {
                Mat capturedFace = new Mat(mRgba, face);
                //Resizing pictures to resolution of Caffe model
                Imgproc.resize(capturedFace, capturedFace, new Size(227, 227));
                //Converting RGBA to BGR
                Imgproc.cvtColor(capturedFace, capturedFace, Imgproc.COLOR_RGBA2BGR);

                //Forwarding picture through Dnn
                Mat inputBlob = Dnn.blobFromImage(capturedFace, 1.0f, new Size(227, 227),
                        new Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);
                mAgeNet.setInput(inputBlob, "data");
                Mat probs = mAgeNet.forward("prob").reshape(1, 1);
                Core.MinMaxLocResult mm = Core.minMaxLoc(probs); //Getting largest softmax output

                double result = mm.maxLoc.x; //Result of age recognition prediction
                Log.i(TAG, "Result is: " + result);
                return AGES[(int) result];
            }
        } catch (Exception e) {
            Log.e(TAG, "Error processing age", e);
        }
        return null;
    }

    private String predictMask (Mat mRgba, Rect[] facesArray) {
        try {
            for (Rect face : facesArray) {
                Mat capturedFace = new Mat(mRgba, face);
                //Resizing pictures to resolution of Caffe model
                Imgproc.resize(capturedFace, capturedFace, new Size(300, 300));
                //Converting RGBA to BGR
                Imgproc.cvtColor(capturedFace, capturedFace, Imgproc.COLOR_RGBA2BGR);

                //Forwarding picture through Dnn
                Mat inputBlob = Dnn.blobFromImage(capturedFace, 1.0f, new Size(300, 300),
                        new Scalar(104.0, 177.0, 123.0), false, false);
                mMaskNet.setInput(inputBlob, "data");
                Mat probs = mMaskNet.forward().reshape(2);

                Core.MinMaxLocResult mm = Core.minMaxLoc(probs); //Getting largest softmax output

                double result = mm.maxLoc.x; //Result of gender recognition prediction. 1 = FEMALE, 0 = MALE
                Log.i(TAG, "Result is: " + result);
                if (result >= 0.6f) {
                    return MASKS[1];
                }
                else {
                    return MASKS[0];
                }

//                return MASKS[(int) result];
            }
        } catch (Exception e) {
            Log.e(TAG, "Error processing mask", e);
        }
        return null;
    }

    //Loading data from assets
    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;

        try {
            //Reading data from app/src/main/assets
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();

            File outputFile = new File(context.getFilesDir(), file);
            FileOutputStream fileOutputStream = new FileOutputStream(outputFile);
            fileOutputStream.write(data);
            fileOutputStream.close();
            return outputFile.getAbsolutePath();
        }catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }
}
