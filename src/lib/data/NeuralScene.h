/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/Core.h"

#include "SceneData.h"
#include "Settings.h"
#include "config.h"
#include "data/NeuralStructure.h"
#include "models/NeuralCamera.h"
#include "models/NeuralTexture.h"
#include "rendering/EnvironmentMap.h"
#include "rendering/NeuralPointCloudCuda.h"


using namespace Saiga;

class NeuralScene
{
   public:
    NeuralScene(std::shared_ptr<SceneData> scene, std::shared_ptr<CombinedParams> params);


    void BuildOutlierCloud(int n);

    void Train(int epoch_id, bool train);

    // 将数据放到CUDA上
    void to(torch::Device device)
    {
        if (environment_map)
        {
            environment_map->to(device);
        }
        texture->to(device);
        camera->to(device);
        intrinsics->to(device);
        poses->to(device);
        point_cloud_cuda->to(device);
        if (outlier_point_cloud_cuda)
        {
            outlier_point_cloud_cuda->to(device);
        }
    }

    // 存储训练结果
    void SaveCheckpoint(const std::string& dir, bool reduced);
    // 加载训练结果
    void LoadCheckpoint(const std::string& dir);

    // 记录（LOG）中间结果
    void Log(const std::string& log_dir);

    void OptimizerStep(int epoch_id, bool structure_only);
    // 对Learning Rate进行更新
    void UpdateLearningRate(int epoch_id , double factor);

    // Download + Save in 'scene'
    // 下载内参到场景
    void DownloadIntrinsics();
    // 下载Pose到场景
    void DownloadPoses();

   public:
    // 友元->可以渲染
    friend class NeuralPipeline;
    // 这里的scene是存储所有的Environmental map
    std::shared_ptr<SceneData> scene;

    NeuralPointCloudCuda point_cloud_cuda         = nullptr;
    NeuralPointCloudCuda outlier_point_cloud_cuda = nullptr;

    NeuralPointTexture texture     = nullptr;
    // 在这里得到Environment Map，环境数据，用来渲染
    EnvironmentMap environment_map = nullptr;
    NeuralCamera camera            = nullptr;
    PoseModule poses               = nullptr;
    // 内参Modual
    IntrinsicsModule intrinsics    = nullptr;
    // 几个优化器，不同模块对应不同优化器
    std::shared_ptr<torch::optim::Optimizer> camera_adam_optimizer, camera_sgd_optimizer;
    std::shared_ptr<torch::optim::Optimizer> texture_optimizer;
    std::shared_ptr<torch::optim::Optimizer> structure_optimizer;
    // GPU设备
    torch::DeviceType device = torch::kCUDA;
    // 这里是参数
    std::shared_ptr<CombinedParams> params;
};
