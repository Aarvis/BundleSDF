#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <iostream>
int main() {
    if (cv::cuda::getCudaEnabledDeviceCount() < 1) return 1;
    cv::Mat a = (cv::Mat_<float>(3, 2) << 0, 0, 1, 1, 2, 2);
    cv::cuda::GpuMat gpu_a(a);
    auto matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> matches;
    matcher->knnMatch(gpu_a, gpu_a, matches, 1);
    if (matches.size() != 3 || matches[0][0].distance != 0) return 2;
    std::cout << "OpenCV " << CV_VERSION << " CUDA descriptor matching passed\n";
}
