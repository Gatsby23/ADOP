/**
* Copyright (c) 2021 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
 */

#pragma once
#include "saiga/vision/torch/TorchHelper.h"

#include "config.h"
#include "data/NeuralStructure.h"
#include "data/SceneData.h"
#include "data/Settings.h"

class EnvironmentMapImpl : public torch::nn::Module
{
   public:
    EnvironmentMapImpl(int channels, int h, int w, bool log_texture);

    EnvironmentMapImpl(torch::Tensor tex);

    // Samples the env. map in all layers and all images of the batch.
    // The result is an array of tensor where each element resebles one layer of the stack.
    // 这里是渲染过程，将Environment map向不同层图像进行投影渲染的过程
    std::vector<torch::Tensor> Sample(torch::Tensor poses, torch::Tensor intrinsics,
                                      ArrayView<ReducedImageInfo> info_batch, int num_layers);


    // [channels, h , w]
    torch::Tensor texture;
};
TORCH_MODULE(EnvironmentMap);
