#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main() {
    const std::string window_name = "QCAP Preview";

    // 和你目前正常的 gst-launch 幾乎同樣邏輯
    std::string pipeline =
        "v4l2src device=/dev/video0 io-mode=mmap do-timestamp=true ! "
        "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
        "videoconvert n-threads=2 ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=1 sync=false";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open QCAP pipeline.\n";
        std::cerr << "Pipeline:\n" << pipeline << std::endl;
        return 1;
    }

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    cv::Mat frame;
    bool first_frame = true;

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Failed to read frame.\n";
            break;
        }

        // 第一次拿到 frame 後，把視窗設成影像原始大小
        // 避免一開始 OpenCV 幫你亂拉伸
        if (first_frame) {
            cv::resizeWindow(window_name, frame.cols, frame.rows);
            first_frame = false;
            std::cout << "Frame size: " << frame.cols << " x " << frame.rows << std::endl;
        }

        cv::imshow(window_name, frame);

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
