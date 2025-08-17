#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo.hpp>
#include <vitis/ai/yolov3.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

int random(int min, int max) {
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

int main(int argc, char *argv[]) {
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

    struct Color {
        int r = 0;
        int g = 0;
        int b = 0;
    };

    Color colors[80] = {{0}};
    srand(time(0));
    for (int j = 0; j < 80; j++) {
        colors[j].r = random(0, 255);
        colors[j].g = random(0, 255);
        colors[j].b = random(0, 255);
    }

    auto yolo = vitis::ai::YOLOv3::create(argv[1], true);
    Mat img = imread(argv[2], IMREAD_COLOR);
    auto results = yolo->run(img);
    for (auto &box : results.bboxes) {
        int label = box.label;
        float xmin = box.x * img.cols;
        float ymin = box.y * img.rows;
        float xmax = xmin + box.width * img.cols;
        float ymax = ymin + box.height * img.rows;
        if (xmin < 1.) xmin = 1.;
        if (ymin < 1.) ymin = 1.;
        if (xmax > img.cols) xmax = img.cols;
        if (ymax > img.rows) ymax = img.rows;
        float confidence = box.score;
        if (label > -1) {
            rectangle(img, Point(xmin, ymin), Point(xmax, ymax),
                Scalar(colors[label].b, colors[label].g, colors[label].r), 2, 2, 0);
            putText(img, classes[label], Point(xmin, ymin - 10),
                cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(colors[label].r, colors[label].g, colors[label].b), 2);
        }
    }

    string filename = argv[2];
    filename += "_detected.jpg";
    imwrite(filename, img);
}
