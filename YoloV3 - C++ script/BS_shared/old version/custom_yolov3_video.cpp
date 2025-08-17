#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo.hpp>
#include <vitis/ai/yolov3.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include "opencv2/opencv.hpp"
#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <chrono>
#include <type_traits>
#include <string>
using namespace std;
using namespace cv;
using namespace std::chrono;

int random(int min, int max)
{
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

int main(int argc, char **argv)
{
    const string classes[80] = {"person",
                                "bicycle", "car", "motorbike", "aeroplane", "bus",
                                "train", "truck", "boat", "traffic light",
                                "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                                "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                "baseball glove", "skateboard", "surfboard", "tennis racket",
                                "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                                "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                                "carrot", "hot dog", "pizza", "donut", "cake",
                                "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
                                "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                                "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    struct Color
    {
        int r = 0;
        int g = 0;
        int b = 0;
    };
    Color colors[80] = {{0}};
    srand(time(0));
    for (int j = 0; j < 80; j++)
    {
        colors[j].r = random(0, 255);
        colors[j].g = random(0, 255);
        colors[j].b = random(0, 255);
    }
    cout << "Creating Yolo model" << endl;
    auto yolo = vitis::ai::YOLOv3::create(argv[1], true);
    cout << "Created Yolo model" << endl;
    VideoCapture cap;
    cout << "Opening camera " << endl;
    if (!cap.open(argv[2]))
    {
        cout << "Cant open camera " << argv[2] << endl;
        return 1;
    }
    if (argv[3])
    {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, atoi(argv[3]));
    }
    if (argv[4])
    {
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, atoi(argv[4]));
    }

    while (1)
    {
        Mat img;
        cap >> img;
        if (img.empty())
            return 0;
        auto start = high_resolution_clock::now();
        auto results = yolo->run(img);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        auto fps = 1000000.0 / duration.count();
        for (auto &box : results.bboxes)
        {
            int label = box.label;
            float xmin = box.x * img.cols;
            float ymin = box.y * img.rows;
            float xmax = xmin + box.width * img.cols;
            float ymax = ymin + box.height * img.rows;
            if (xmin < 0.)
                xmin = 1.;
            if (ymin < 0.)
                ymin = 1.;
            if (xmax > img.cols)
                xmax = img.cols;
            if (ymax > img.rows)
                ymax = img.rows;
            float confidence = box.score;
            if (label > -1)
            {
                rectangle(img, Point(xmin, ymin), Point(xmax, ymax),
                          Scalar(colors[label].b, colors[label].g, colors[label].r), 2, 2, 0);
                putText(img, classes[label], Point(xmin, ymin - 10),
                        cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(colors[label].r, colors[label].g, colors[label].b), 2);
            }
            string frame = to_string(fps);
            frame += " FPS";
            putText(img, frame, Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 0, 0), 2);
        }
        imshow("frame", img);
        waitKey(1);
    }
}