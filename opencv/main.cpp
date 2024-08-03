#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0); // Open the default camera (change to file path if using video file)
    
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::cuda::GpuMat d_prev_frame, d_curr_frame, d_flow;
    cv::Mat prev_frame, curr_frame, flow;
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn = cv::cuda::FarnebackOpticalFlow::create();
    cv::namedWindow("Optical Flow", cv::WINDOW_NORMAL);

    cap >> prev_frame;
    d_prev_frame.upload(prev_frame);
    cv::cuda::cvtColor(d_prev_frame, d_prev_frame, cv::COLOR_BGR2GRAY);

    while (true) {
        cap >> curr_frame;
        if (curr_frame.empty())
            break;

        d_curr_frame.upload(curr_frame);
        cv::cuda::cvtColor(d_curr_frame, d_curr_frame, cv::COLOR_BGR2GRAY);

        // Calculate optical flow
        farn->calc(d_prev_frame, d_curr_frame, d_flow);

        // Download flow to CPU for visualization
        d_flow.download(flow);

        // Visualize the optical flow
        cv::Mat flow_parts[2];
        cv::split(flow, flow_parts);
        cv::Mat magnitude, angle;
        cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle);

        // Normalize magnitude to 0-255 range
        cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
        magnitude.convertTo(magnitude, CV_8U);

        // Apply color map
        cv::Mat colored;
        cv::applyColorMap(magnitude, colored, cv::COLORMAP_JET);

        cv::imshow("Optical Flow", colored);

        if (cv::waitKey(30) >= 0)
            break;

        d_prev_frame = d_curr_frame;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
