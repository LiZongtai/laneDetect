package com.example.lanedetect.opencv;

import android.graphics.Bitmap;
import android.os.Build;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.example.lanedetect.utils.OpencvUtils.average;
import static com.example.lanedetect.utils.OpencvUtils.bitmapToMat;
import static com.example.lanedetect.utils.OpencvUtils.calculate_histogram;
import static com.example.lanedetect.utils.OpencvUtils.histogram_peak;
import static com.example.lanedetect.utils.OpencvUtils.mag_thresh;
import static com.example.lanedetect.utils.OpencvUtils.matToByteBuffer;
import static com.example.lanedetect.utils.OpencvUtils.polynomial_curve_fit;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2HLS;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;

public class LaneDetect {
        private static String TAG = "LaneDetect";
        // Original camera image
        private Bitmap orig_frame;
        // This will hold the image after perspective transformation
        private Mat warped_frame;
        private Mat transformation_matrix;
        private Mat inv_transformation_matrix;
        // (Width, Height) of the original video frame (or image)
        private Size size;
        private int width;
        private int height;
        // Four corners of the trapezoid-shaped region of interest, manually.
        private Mat roi_points;
        // The desired corner locations  of the region of interest
        // after we perform perspective transformation.
        private int padding;
        private Mat desired_roi_points;
        // Histogram that shows the white pixel peaks for lane line detection
        private Mat histogram;
        // Sliding window parameters
        private int no_of_windows = 10;
        private int margin;
        private int minpix;
        // Pixel parameters for x and y dimensions
        private float YM_PER_PIX = 10.0f / 1000f; // meters per pixel in y dimension
        private float XM_PER_PIX = 3.7f / 781f; // meters per pixel in x dimension
        // polynomial curve fitting results
        private ArrayList<Double> left_fit;
        private ArrayList<Double> right_fit;
        private int[] left_fitx;
        private int[] right_fitx;
        private int[] left_fity;
        private int[] right_fity;
        public Point[] points;
        // Radii of curvature and offset
        private float left_curvem;
        private float right_curvem;
        private float center_offset;
        private boolean isLandscape = true;

        /**
         * Constructor
         *
         * @param orig_frame  input image
         * @param isLandscape the Device is Landscape
         */
        public LaneDetect(Bitmap orig_frame, boolean isLandscape) {
            this.orig_frame = orig_frame;
            this.isLandscape = isLandscape;
            System.out.println("orig_frame.getHeight:" + orig_frame.getHeight());
            this.width = orig_frame.getWidth();
            this.height = orig_frame.getHeight();
            this.padding = (int) (this.width * 0.25f);// padding from side of the image in pixels
            this.margin = (int) ((1f / 12f) * this.width);
            this.minpix = (int) ((1f / 24f) * this.width);
            this.left_fit = new ArrayList<>();
            this.right_fit = new ArrayList<>();
            this.left_fitx = new int[this.height];
            this.right_fitx = new int[this.height];
            this.left_fity = new int[this.height];
            this.right_fity = new int[this.height];
        }

        public Mat laneDetection() {
            ArrayList<Point> roi_points = get_roi_points((float) this.width, (float) this.height);
            ArrayList<Point> desired_roi_points = get_desired_roi_points(this.width, this.height, this.padding);
            // Perform thresholding to isolate lane lines
            Mat lane_line_markings = get_line_markings(this.orig_frame);
            Mat segment=segment(lane_line_markings);
            // Perform the perspective transform to generate a bird's eye view
            Mat warped_frame = perspective_transform(segment, roi_points, desired_roi_points);
            // Generate the image histogram to serve as a starting point
            // for finding lane line pixels
            byte[] byteBuffer = matToByteBuffer(warped_frame);
            Mat sliding_windows_mat = get_lane_line_indices_sliding_windows(warped_frame);
            get_lane_points();
            return  sliding_windows_mat;
        }

        public Mat getLanePlot() {
            Mat mat = bitmapToMat(this.orig_frame);
            ArrayList<MatOfPoint> polygons = new ArrayList<>();
            polygons.clear();
            if (this.points != null) {
                polygons.add(new MatOfPoint(this.points));
                Imgproc.fillPoly(mat, polygons, new Scalar(0, 0, 255));
            }
            return mat;
        }


        private void get_lane_points() {
            double a00 = this.inv_transformation_matrix.get(0, 0)[0];
            double a01 = this.inv_transformation_matrix.get(0, 1)[0];
            double a02 = this.inv_transformation_matrix.get(0, 2)[0];
            double a10 = this.inv_transformation_matrix.get(1, 0)[0];
            double a11 = this.inv_transformation_matrix.get(1, 1)[0];
            double a12 = this.inv_transformation_matrix.get(1, 2)[0];
            double a20 = this.inv_transformation_matrix.get(2, 0)[0];
            double a21 = this.inv_transformation_matrix.get(2, 1)[0];
            double a22 = this.inv_transformation_matrix.get(2, 2)[0];

            int[] y = new int[this.height];
            for (int i = 0; i < this.height; i++) {
                y[i] = i;
            }
            if (this.left_fit != null && this.right_fit != null) {
                for (int i = 0; i < this.height; i++) {
                    int left_x_temp = (int) (this.left_fit.get(2) * y[i] * y[i] + this.left_fit.get(1) * y[i] + left_fit.get(0));
                    int right_x_temp = (int) (this.right_fit.get(2) * y[i] * y[i] + this.right_fit.get(1) * y[i] + right_fit.get(0));
                    double v_left = a20 * left_x_temp + a21 * i + a22;
                    this.left_fitx[i] = (int) ((a00 * left_x_temp + a01 * i + a02) / v_left);
                    this.left_fity[i] = (int) ((a10 * left_x_temp + a11 * i + a12) / v_left);
                    double v_right = a20 * right_x_temp + a21 * i + a22;
                    this.right_fitx[i] = (int) ((a00 * right_x_temp + a01 * i + a02) / v_right);
                    this.right_fity[i] = (int) ((a10 * right_x_temp + a11 * i + a12) / v_right);
                }
            }
            this.points = new Point[this.height * 2];
            for (int i = 0; i < this.height; i++) {
                this.points[2 * i] = new Point(this.left_fitx[i], this.left_fity[i]);
                this.points[2 * i + 1] = new Point(this.right_fitx[i], this.right_fity[i]);
            }
//        System.out.println("transformation_matrix:"+transformation_matrix.size());
        }

        private Mat segment(Mat frame){
            Mat segment = new Mat();
            Mat mask = new Mat(frame.size(), frame.type(), Scalar.all(0));
            ArrayList<MatOfPoint> polygons = new ArrayList<>();
//        polygons.add(new MatOfPoint(0, image_canny.height()));
//        polygons.add(new MatOfPoint(image_canny.width(), image_canny.height()));
//        polygons.add(new MatOfPoint((int) (image_canny.width() / 2), (int) (image_canny.height() / 2)));
            Point[] points = new Point[6];
            points[0] = new Point(0,this.height);
            points[1] = new Point(this.width,this.height);
            points[2] = new Point(this.width,this.height*0.75);
            points[3] = new Point(this.width*0.6,this.height*0.5);
            points[4] = new Point(this.width*0.4,this.height*0.5);
            points[5] = new Point(0,this.height*0.75);
//        points[0] = new Point((int) (frame.width() / 2), (int) (frame.height() / 2));
//        points[1] = new Point(50,frame.height());
//        points[2] = new Point(frame.width()-50, frame.height());
            polygons.clear();
            polygons.add(new MatOfPoint(points));
            Imgproc.fillPoly(mask, polygons, new Scalar(255, 255, 255));
            Core.bitwise_and(frame, mask, segment);
            return segment;
        }

        private Mat perspective_transform(Mat frame, ArrayList<Point> roi_points, ArrayList<Point> desired_roi_points) {
            Mat roi = Converters.vector_Point2f_to_Mat(roi_points);
            Mat desired_roi = Converters.vector_Point2f_to_Mat(desired_roi_points);
            // Calculate the transformation matrix
            this.transformation_matrix = Imgproc.getPerspectiveTransform(roi, desired_roi);
            // Calculate the inverse transformation matrix
            this.inv_transformation_matrix = Imgproc.getPerspectiveTransform(desired_roi, roi);
            // Perform the transform using the transformation matrix
            Mat warped_frame = new Mat();
            Imgproc.warpPerspective(frame, warped_frame, this.transformation_matrix, frame.size(), INTER_LINEAR);
            // Convert image to binary
            Mat binary_warped = new Mat();
            Imgproc.threshold(warped_frame, binary_warped, 127, 255, THRESH_BINARY);
            return binary_warped;
        }

        /**
         * Four corners of the trapezoid-shaped region of interest (ROI)
         *
         * @param w width
         * @param h height
         * @return roi points array int[][]
         */
        private static ArrayList<Point> get_roi_points(float w, float h) {
            float width_top = 0.4f;
            float width_bottom = 0.04f;
            float height_top = 0.5f;
            ArrayList<Point> roi_arr = new ArrayList<>();
            roi_arr.add(new Point((float) (w * width_top), (float) (h * height_top)));  // Top-left corner
            roi_arr.add(new Point((float) (w * width_bottom), (float) h - 1));            // Bottom-left corner
            roi_arr.add(new Point((float) (w * (1f - width_bottom)), (float) h - 1));     // Bottom-right corner
            roi_arr.add(new Point((float) (w * (1f - width_top)), (float) (h * height_top)));// Top-right corner
            return roi_arr;
        }

        /**
         * The desired corner locations  of the region of interest after performing perspective transformation
         *
         * @param w width
         * @param h height
         * @param p padding
         * @return desired_roi_points array int[][]
         */
        private static ArrayList<Point> get_desired_roi_points(float w, float h, float p) {
            ArrayList<Point> desired_roi_arr = new ArrayList<>();
            desired_roi_arr.add(new Point(p, 0));  // Top-left corner
            desired_roi_arr.add(new Point(p, h));            // Bottom-left corner
            desired_roi_arr.add(new Point(w - p, h));     // Bottom-right corner
            desired_roi_arr.add(new Point(w - p, 0));// Top-right corner
            return desired_roi_arr;
        }

        /**
         * Isolates lane lines.
         *
         * @param image The camera frame that contains the lanes we want to detect
         * @return Binary (i.e. black and white) image containing the lane lines.
         */
        private static Mat get_line_markings(Bitmap image) {
        /*
            Convert the video frame from BGR (blue, green, red)
            color space to HLS (hue, saturation, lightness).
         */
            Mat frame = bitmapToMat(image);
            Mat hls = new Mat();
            Mat channel = new Mat();
            Mat sxbinary = new Mat();
            Imgproc.cvtColor(frame, hls, COLOR_BGR2HLS);
        /*
            ################### Isolate possible lane line edges ######################
            Perform Sobel edge detection on the L (lightness) channel of
            the image to detect sharp discontinuities in the pixel intensities
            along the x and y axis of the video frame.
            sxbinary is a matrix full of 0s (black) and 255 (white) intensity values
            Relatively light pixels get made white. Dark pixels get made black.
         */
            int[] thresh = {120, 255};
            Core.extractChannel(hls, channel, 1);
            Imgproc.threshold(channel, sxbinary, thresh[0], thresh[1], THRESH_BINARY);
            Imgproc.GaussianBlur(sxbinary, sxbinary, new Size(3, 3), 0);
        /*
            1s will be in the cells with the highest Sobel derivative values
            (i.e. strongest lane line edges)
         */
            Mat binary = mag_thresh(sxbinary, 3, new int[]{110, 1});
        /*
            ######################## Isolate possible lane lines ######################
            Perform binary thresholding on the S (saturation) channel
            of the video frame. A high saturation value means the hue color is pure.
            We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
            and have high saturation channel values.
            s_binary is matrix full of 0s (black) and 255 (white) intensity values
            White in the regions with the purest hue colors (e.g. >80...play with
            this value for best results).
         */
            Mat s_channel = new Mat();
            Mat s_binary = new Mat();
            Core.extractChannel(hls, s_channel, 2);  // use only the saturation channel data
            thresh = new int[]{80, 255};
            Imgproc.threshold(s_channel, s_binary, thresh[0], thresh[1], THRESH_BINARY);
        /*
            Perform binary thresholding on the R (red) channel of the
            original BGR video frame.
            r_thresh is a matrix full of 0s (black) and 255 (white) intensity values
            White in the regions with the richest red channel values (e.g. >120).
            Remember, pure white is bgr(255, 255, 255).
            Pure yellow is bgr(0, 255, 255). Both have high red channel values.
         */
            thresh = new int[]{120, 255};
            Mat r_channel = new Mat();
            Mat r_thresh = new Mat();
            Core.extractChannel(frame, r_channel, 2);
            Imgproc.threshold(r_channel, r_thresh, thresh[0], thresh[1], THRESH_BINARY);
        /*
            Lane lines should be pure in color and have high red channel values
            Bitwise AND operation to reduce noise and black-out any pixels that
            don't appear to be nice, pure, solid colors (like white or yellow lane
            lines.)
         */
            Mat rs_binary = new Mat();
            Core.bitwise_and(s_binary, r_thresh, rs_binary);
        /*
            Combine the possible lane lines with the possible lane line edges #####
            If you show rs_binary visually, you'll see that it is not that different
            from this return value. The edges of lane lines are thin lines of pixels.
         */
            Mat lane_line_markings = new Mat();
            Core.bitwise_or(rs_binary, binary, lane_line_markings);
            return lane_line_markings;
        }

        private Mat get_lane_line_indices_sliding_windows(Mat warped_frame) {
            // Sliding window width is +/- margin
            int margin = this.margin;
            Mat frame_sliding_window = new Mat();
            warped_frame.copyTo(frame_sliding_window);
            // Set the height of the sliding windows
            int window_height = (int) (warped_frame.size().height / this.no_of_windows);
            // Find the x and y coordinates of all the nonzero
            ArrayList<Integer> nonzerox = new ArrayList<Integer>();
            ArrayList<Integer> nonzeroy = new ArrayList<Integer>();
            int size = (int) warped_frame.total();
            //conver to byteBuffer for faster traveling
            byte[] byteBuffer = matToByteBuffer(warped_frame);
            for (int i = 0; i < size; i++) {
                if (byteBuffer[i] != 0) {
                    nonzeroy.add(i / warped_frame.cols());
                    nonzerox.add(i % warped_frame.cols());
                }
            }
            // Store the pixel indices for the left and right lane lines
            ArrayList<Integer> left_lane_inds = new ArrayList<Integer>();
            ArrayList<Integer> right_lane_inds = new ArrayList<Integer>();
            // Current positions for pixel indices for each window,
            // which we will continue to update
            // hough变换
//        Mat hough = new Mat();
//        Imgproc.HoughLinesP(warped_frame, hough, 2, Math.PI / 180, 50, 50, 50);
            int[] histogram = calculate_histogram(warped_frame);
            int[] base = histogram_peak(histogram);
//        int[] base = calculate_lines(hough);
            System.out.println("hough bases:" + Arrays.toString(base));
            int leftx_current = base[0];
            int rightx_current = base[1];
            // Go through one window at a time
            for (int i = 0; i < this.no_of_windows; i++) {
                // Identify window boundaries in x and y (and right and left)
                int win_y_low = this.height - (i + 1) * window_height;
                int win_y_high = this.height - i * window_height;
                int win_xleft_low = leftx_current - margin;
                int win_xleft_high = leftx_current + margin;
                int win_xright_low = rightx_current - margin;
                int win_xright_high = rightx_current + margin;
                Rect rect_left = new Rect(new Point(win_xleft_low, win_y_low), new Point(win_xleft_high, win_y_high));
                Rect rect_right = new Rect(new Point(win_xright_low, win_y_low), new Point(win_xright_high, win_y_high));
                Imgproc.rectangle(frame_sliding_window, rect_left, new Scalar(255, 255, 255), 2);
                Imgproc.rectangle(frame_sliding_window, rect_right, new Scalar(255, 255, 255), 2);
                // Identify the nonzero pixels in x and y within the window
                List<Integer> good_left_inds = new ArrayList<>();
                List<Integer> good_right_inds = new ArrayList<>();
                int len = nonzerox.size();
                for (int j = 0; j < len; j++) {
                    int nx = nonzerox.get(j);
                    int ny = nonzeroy.get(j);
                    if (ny >= win_y_low && ny < win_y_high) {
                        if (nx >= win_xleft_low && nx < win_xleft_high) {
                            good_left_inds.add(j);
                            left_lane_inds.add(j);
                        } else if (nx >= win_xright_low && nx < win_xright_high) {
                            good_right_inds.add(j);
                            right_lane_inds.add(j);
                        }
                    }
                }

//                if (good_left_inds.size() > this.minpix) {
//                    leftx_current = average(nonzerox.subList(Collections.min(good_left_inds), Collections.max(good_left_inds)));
//                }
//                if (good_right_inds.size() > this.minpix) {
//                    rightx_current = average(nonzerox.subList(Collections.min(good_right_inds), Collections.max(good_right_inds)));
//                }
            }
            // Extract the pixel coordinates for the left and right lane lines
            int len_left = left_lane_inds.size();
            int len_right = right_lane_inds.size();
            System.out.println("left_lane_inds.size:" + len_left + "right_lane_inds.size:" + len_right);
            ArrayList<Integer> leftx = new ArrayList<>();
            ArrayList<Integer> lefty = new ArrayList<>();
            ArrayList<Integer> rightx = new ArrayList<>();
            ArrayList<Integer> righty = new ArrayList<>();
            for (int i = 0; i < len_left; i++) {
                leftx.add(nonzerox.get(left_lane_inds.get(i)));
                lefty.add(nonzeroy.get(left_lane_inds.get(i)));
            }
            for (int i = 0; i < len_right; i++) {
                rightx.add(nonzerox.get(right_lane_inds.get(i)));
                righty.add(nonzeroy.get(right_lane_inds.get(i)));
            }
            // Fit a second order polynomial curve to the pixel coordinates for the left and right lane lines
            System.out.println("fit leftx: " + leftx.toString());
            System.out.println("fit rightx: " + rightx.toString());
            //fit x=ay^2+by+c
            this.left_fit = polynomial_curve_fit(leftx, lefty, 2);
            this.right_fit = polynomial_curve_fit(rightx, righty, 2);
            System.out.println("left_fit: " + this.left_fit.toString());
            System.out.println("right_fit: " + this.right_fit.toString());
            return frame_sliding_window;
            // Create the x and y values to plot on the image
        }

        private int[] calculate_lines(Mat lines) {
            ArrayList<Integer> points_x_right = new ArrayList<>();
            ArrayList<Integer> points_x_left=new ArrayList<>();
            if (lines.cols() > 0) {
                for (int i = 0; i < lines.rows(); i++) {
                    double[] l = lines.get(i, 0);
//                System.out.println("l: " + Arrays.toString(l));
                    if (Math.cos((l[1] - l[3]) / (l[0] - l[2])) > 0.5d) {
//                    Point p1 = new Point(l[0], l[1]);
//                    Point p2 = new Point(l[2], l[3]);
//                System.out.println("p1: " + p1.x + "," + p1.y + " || p2: " + p2.x + "," + p2.y);
                        points_x_left.add((int)((l[0]+l[2])/2));
                    }else if(Math.cos((l[1] - l[3]) / (l[0] - l[2])) < -0.5d){
                        points_x_right.add((int)((l[0]+l[2])/2));
                    }
                }
            }
            int[] base = new int[2];
            int left_len=points_x_left.size();
            int right_len=points_x_right.size();
//        System.out.println("points x left length: " + points_x_left.size());
//        System.out.println("points x right length: " + points_x_right.size());
            int sum_left=0;
            int sum_right=0;
            if(left_len>0){
                for(int i = 0; i < points_x_left.size(); ++i)
                {
                    sum_left = points_x_left.get(i)+sum_left;
                }
                //求平均数
                base[0] = sum_left/points_x_left.size();
            }else{
                base[0] = (int)(this.width*0.4);
            }
            if(right_len>0){
                for(int i = 0; i < points_x_right.size(); ++i)
                {
                    sum_right = points_x_right.get(i)+sum_right;
                }
                base[1] = sum_right/points_x_right.size();
            }else{
                base[0] = (int)(this.width*0.6);
            }

            return base;
        }
}