// PersonDetector 단독 데모: 이미지/영상/카메라 입력에 검출 박스를 그려 보여준다.
//
// 사용법:
//   person_detector_demo <model.onnx> <이미지|영상 경로|카메라 번호> [옵션]
// 옵션:
//   --conf <값>     objectness 임계값 (기본 0.25, RPi4 배포 기본값과 동일)
//   --nms <값>      NMS IoU 임계값 (기본 0.3)
//   --save <경로>   결과 저장 (이미지 입력이면 이미지, 영상 입력이면 mp4)
//   --no-display    창을 띄우지 않음 (헤드리스 환경/자동 테스트용)
//
// 예: person_detector_demo ../models/person_detector_lite.onnx test.jpg --save out.jpg

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "PersonDetector.h"

namespace {

bool hasImageExtension(std::string path) {
    std::transform(path.begin(), path.end(), path.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const char* ext : {".jpg", ".jpeg", ".png", ".bmp"}) {
        const std::string e(ext);
        if (path.size() >= e.size() && path.compare(path.size() - e.size(), e.size(), e) == 0) {
            return true;
        }
    }
    return false;
}

bool isNumber(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
}

void printUsage(const char* prog) {
    std::cerr << "usage: " << prog
              << " <model.onnx> <image|video|camera-index> [--conf 0.25] [--nms 0.3]"
                 " [--save <path>] [--no-display]\n";
}

void printDetections(const std::vector<detection::DetectionBox>& boxes) {
    auto feet = detection::PersonDetector::getFootPoints(boxes);
    std::cout << "detections: " << boxes.size() << "\n";
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& b = boxes[i];
        std::cout << "  [" << i << "] box=(" << b.xmin << ", " << b.ymin << ", " << b.xmax << ", "
                  << b.ymax << ") conf=" << b.conf << " foot=(" << feet[i].x << ", " << feet[i].y << ")\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string modelPath = argv[1];
    const std::string input = argv[2];
    float conf = 0.25f;
    float nms = 0.3f;
    std::string savePath;
    bool display = true;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--conf" && i + 1 < argc) {
            conf = std::stof(argv[++i]);
        } else if (a == "--nms" && i + 1 < argc) {
            nms = std::stof(argv[++i]);
        } else if (a == "--save" && i + 1 < argc) {
            savePath = argv[++i];
        } else if (a == "--no-display") {
            display = false;
        } else {
            printUsage(argv[0]);
            return 1;
        }
    }

    // 실험 11(경량 custom 백본)은 320x240 입력으로 학습됨
    detection::PersonDetector detector(modelPath, 320, 240, conf, nms);
    if (!detector.isLoaded()) {
        std::cerr << "failed to load model: " << modelPath << "\n";
        return 1;
    }
    std::cout << "model: " << modelPath << " (conf=" << conf << ", nms=" << nms << ")\n";

    if (hasImageExtension(input)) {
        cv::Mat frame = cv::imread(input);
        if (frame.empty()) {
            std::cerr << "failed to read image: " << input << "\n";
            return 1;
        }
        auto boxes = detector.detect(frame, /*drawBoxes=*/true);
        printDetections(boxes);
        if (!savePath.empty()) {
            cv::imwrite(savePath, frame);
            std::cout << "saved: " << savePath << "\n";
        }
        if (display) {
            cv::imshow("PersonDetector demo (any key to quit)", frame);
            cv::waitKey(0);
        }
        return 0;
    }

    cv::VideoCapture cap;
    if (isNumber(input)) {
        cap.open(std::stoi(input));
    } else {
        cap.open(input);
    }
    if (!cap.isOpened()) {
        std::cerr << "failed to open video/camera: " << input << "\n";
        return 1;
    }

    cv::VideoWriter writer;
    int frames = 0;
    double inferSeconds = 0.0;
    cv::Mat frame;
    while (cap.read(frame)) {
        if (!savePath.empty() && !writer.isOpened()) {
            double fps = cap.get(cv::CAP_PROP_FPS);
            writer.open(savePath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps > 0 ? fps : 20.0,
                        frame.size());
        }

        auto t0 = std::chrono::steady_clock::now();
        auto boxes = detector.detect(frame, /*drawBoxes=*/true);
        inferSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++frames;

        cv::putText(frame, "persons: " + std::to_string(boxes.size()), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        if (writer.isOpened()) {
            writer.write(frame);
        }
        if (display) {
            cv::imshow("PersonDetector demo (ESC/q to quit)", frame);
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') {
                break;
            }
        }
    }

    std::cout << "frames: " << frames;
    if (frames > 0) {
        std::cout << ", avg detect time: " << (inferSeconds / frames * 1000.0) << " ms ("
                  << (frames / inferSeconds) << " FPS)";
    }
    std::cout << "\n";
    return 0;
}
