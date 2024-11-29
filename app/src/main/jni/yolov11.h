#ifndef NCNN_ANDROID_YOLOV11_YOLOV11_H
#define NCNN_ANDROID_YOLOV11_YOLOV11_H

#include <vector>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <net.h>


struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
};


class Inference
{
public:
    Inference();
    int loadNcnnNetwork(AAssetManager* mgr, const char* modeltype , const int& modelInputShape, const float* meanVals, const float* normVals, bool useGpu = false);
    std::vector<Object> runInference(const cv::Mat &input);
    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

private:
    ncnn::Net net;

    std::string modelPath{};
    bool gpuEnabled{};

    int target_size;


    float meanVals[3];
    float normVals[3];

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif //NCNN_ANDROID_YOLOV11_YOLOV11_H
