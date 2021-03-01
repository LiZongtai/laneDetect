package com.example.lanedetect.utils;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.opencv.core.CvType.CV_64F;
import static org.opencv.core.CvType.CV_64FC1;
import static org.opencv.core.CvType.CV_8U;

public class OpencvUtils {

    public static byte[] matToByteBuffer(Mat mat){
        byte[] byteBuffer = new byte[(int)mat.total()];
        mat.get(0, 0, byteBuffer);
        return byteBuffer;
    }

    /**
     * convert Mat to Bitmap
     *
     * @param mat Mat
     * @return Bitmap
     */
    public static Bitmap matToBitmap(Mat mat) {
        Bitmap resultBitmap = null;
        if (mat != null) {
            resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            if (resultBitmap != null)
                Utils.matToBitmap(mat, resultBitmap);
        }
        return resultBitmap;
    }

    //

    /**
     * convert Bitmap to Mat
     *
     * @param bm Bitmap
     * @return Mat
     */
    public static Mat bitmapToMat(Bitmap bm) {
        Bitmap bmp32 = bm.copy(Bitmap.Config.RGB_565, true);
        Mat imgMat = new Mat(bm.getHeight(), bm.getWidth(), CvType.CV_8UC2, new Scalar(0));
        Utils.bitmapToMat(bmp32, imgMat);
        return imgMat;

    }

    /**
     * Find edges that are aligned vertically and horizontally on the image
     *
     * @param img_channel Channel from an image
     * @return Image with Sobel edge detection applied
     */
    public static Mat sobel(Mat img_channel) {
        Mat sobel = new Mat();
        int sobel_kernel = 3;
        // Will detect differences in pixel intensities going from
        // left to right on the image (i.e. edges that are vertically aligned)
        Imgproc.Sobel(img_channel, sobel, CV_64F, 1, 0, sobel_kernel);
        return sobel;
    }

    /**
     * Find edges that are aligned vertically and horizontally on the image
     *
     * @param img_channel Channel from an image
     * @param orient      Across which axis of the image are we detecting edges?
     * @return Image with Sobel edge detection applied
     */
    public static Mat sobel(Mat img_channel, boolean orient) {
        Mat sobel = new Mat();
        int sobel_kernel = 3;
        if (orient) {
            // Will detect differences in pixel intensities going from
            // left to right on the image (i.e. edges that are vertically aligned)
            Imgproc.Sobel(img_channel, sobel, CV_64F, 1, 0, sobel_kernel);
        }
        if (!orient) {
            // Will detect differences in pixel intensities going from
            // top to bottom on the image (i.e. edges that are horizontally aligned)
            Imgproc.Sobel(img_channel, sobel, CV_64F, 0, 1, sobel_kernel);
        }
        return sobel;
    }

    /**
     * Find edges that are aligned vertically and horizontally on the image
     *
     * @param img_channel  Channel from an image
     * @param orient       Across which axis of the image are we detecting edges?
     * @param sobel_kernel No. of rows and columns of the kernel (i.e. 3x3 small matrix)
     * @return Image with Sobel edge detection applied
     */
    public static Mat sobel(Mat img_channel, boolean orient, int sobel_kernel) {
        Mat sobel = new Mat();
        if (orient) {
            // Will detect differences in pixel intensities going from
            // left to right on the image (i.e. edges that are vertically aligned)
            Imgproc.Sobel(img_channel, sobel, CV_64F, 1, 0, sobel_kernel);
        }
        if (!orient) {
            // Will detect differences in pixel intensities going from
            // top to bottom on the image (i.e. edges that are horizontally aligned)
            Imgproc.Sobel(img_channel, sobel, CV_64F, 0, 1, sobel_kernel);
        }
        return sobel;
    }

    /**
     * Implementation of Sobel edge detection
     * @param image  2D or 3D array to be blurred
     * @param sobel_kernel Size of the small matrix (i.e. kernel)
     * @param thresh
     * @return Binary (black and white) 2D mask image
     */
    public static Mat mag_thresh(Mat image, int sobel_kernel, int[] thresh) {
        // Get the magnitude of the edges that are vertically aligned on the image
        Mat sobelx_mat = sobel(image, true, sobel_kernel);
        Mat sobelx_zeros= Mat.zeros(sobelx_mat.size(),sobelx_mat.type());
        Mat sobelx = new Mat();
        Core.absdiff(sobelx_mat,sobelx_zeros,sobelx);
        // Get the magnitude of the edges that are horizontally aligned on the image
        Mat sobely_mat = sobel(image, false, sobel_kernel);
        Mat sobely=new Mat();
        Core.absdiff(sobely_mat,sobelx_zeros,sobely);
        // Find areas of the image that have the strongest pixel intensity changes
        // in both the x and y directions. These have the strongest gradients and
        // represent the strongest edges in the image (i.e. potential lane lines)
        // mag is a 2D array .. number of rows x number of columns = number of pixels
        // from top to bottom x number of pixels from left to right
        Mat mag_add=new Mat();
        Mat mag_pow_x=new Mat();
        Mat mag_pow_y=new Mat();
        Mat mag=new Mat();
        Core.pow(sobelx,2,mag_pow_x);
        Core.pow(sobely,2,mag_pow_y);
        Core.add(mag_pow_x,mag_pow_y,mag_add);
        Core.sqrt(mag_add,mag);
        // Return a 2D array that contains 0s and 1s
        Mat binary = new Mat();
        // If value == 0, make all values in binary equal to 0 if the
        // corresponding value in the input array is between the threshold
        // (inclusive). Otherwise, the value remains as 1. Therefore, the pixels
        // with the high Sobel derivative values (i.e. sharp pixel intensity
        // discontinuities) will have 0 in the corresponding cell of binary.
//        Mat thresh_top = new Mat(mag.type()).setTo(new Scalar(thresh[0]));
//        Mat thresh_bottom = new Mat(mag.type()).setTo(new Scalar(thresh[1]));
//        Mat bool_arr=new Mat();
        Imgproc.threshold(mag,binary,thresh[0],(int)thresh[1], Imgproc.THRESH_BINARY_INV);
//        for (int i = 0; i < binary.rows(); i++) {
//            for (int j = 0; j < binary.cols(); j++) {
//                if ((int)bool_arr.get(i,j)[0]!=0) {
//                    binary.get(i,j)[0]=0;
//                }
//            }
//        }
        //convert Mat type from CV_64F to CV_9U
        binary.convertTo(binary,CV_8U);
        return binary;
    }

    /**
     * Calculate the image histogram to find peaks in white pixel count
     * @param frame The warped image
     * @return int[] histogram
     */
    public static int[] calculate_histogram(Mat frame) {
        // Generate the histogram
        int[] hist=new int[frame.cols()];
        for(int i=0;i<frame.cols();i++){
            Scalar sum= Core.sumElems(frame.col(i));
            hist[i]= (int)sum.val[0];
        }
        return hist;
    }

    public static int[] histogram_peak(int[] histogram){
        int midpoint =(int)histogram.length/2;
        int leftx_base=getMaxIndex(Arrays.copyOfRange(histogram,0,midpoint));
        int rightx_base=getMaxIndex(Arrays.copyOfRange(histogram,midpoint,histogram.length-1))+midpoint;
        // x coordinate of left peak, x coordinate of right peak
        return new int[]{leftx_base, rightx_base};
    }
    public static int getMaxIndex(int[] arr) {
        if(arr==null||arr.length==0){
            return 0;
        }
        int maxIndex=0;
        for(int i =0;i<arr.length-1;i++){
            if(arr[maxIndex]<arr[i+1]){
                maxIndex=i+1;
            }
        }
        return maxIndex;
    }

    public static int average(List<Integer> list) {
        // 'average' is undefined if there are no elements in the list.
        if (list == null || list.isEmpty())
            return 0;
        // Calculate the summation of the elements in the list
        long sum = 0;
        int n = list.size();
        // Iterating manually is faster than using an enhanced for loop.
        for (int i = 0; i < n; i++)
            sum += list.get(i);
        // We don't want to perform an integer division, so the cast is mandatory.
        return ((int) sum) / n;
    }

    /**
     * polynomial curve fitting
     * @param ylist y list
     * @param xlist x list
     * @param n n
     * @return fit list
     */
    public static ArrayList<Double> polynomial_curve_fit(ArrayList<Integer> ylist, ArrayList<Integer> xlist, int n){
        //Number of key points
        int N=xlist.size();
        double[] X=new double[(n+1)*(n+1)];
        // Matrix X
        for(int i=0;i<n+1;i++){
            for (int j=0;j<n+1;j++){
                int ij=i*(n+1)+j;
                X[ij]=0;
                for(int k=0;k<N;k++){
                    X[ij]=X[ij]+(Math.pow(xlist.get(k),i+j));
                }
            }
        }
        Mat X_mat=new Mat(n+1,n+1,CV_64FC1);
        X_mat.put(0,0,X);
//        Mat X_mat_temp= Converters.vector_double_to_Mat(X);
//        X_mat_temp.convertTo(X_mat,CV_64FC1);
//        X_mat=X_mat.reshape(0,n+1);
        // Matrix Y
        double[] Y=new double[n+1];
        for(int i=0;i<n+1;i++){
            Y[i]=0;
            for (int k=0;k<N;k++){
                Y[i]=Y[i]+Math.pow(xlist.get(k),i)*ylist.get(k);
            }
        }
        Mat Y_mat=new Mat(n + 1, 1,CV_64FC1);
        Y_mat.put(0,0,Y);
//        Mat Y_mat_temp= Converters.vector_double_to_Mat(Y);
//        Y_mat_temp.convertTo(Y_mat,CV_64FC1);
        // solve Matrix A
        ArrayList<Double> solve=new ArrayList<>();
        Mat solve_mat=new Mat(n + 1, 1, CV_64FC1);
        if(!X_mat.empty() && !Y_mat.empty()){
            Core.solve(X_mat,Y_mat,solve_mat, Core.DECOMP_LU);
        }
        Converters.Mat_to_vector_double(solve_mat,solve);
        return solve;
    }
}
