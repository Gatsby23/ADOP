/**
* Copyright (c) 2021 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
*/

#include "RenderModule.h"

#include <torch/torch.h>


PointRenderModuleImpl::PointRenderModuleImpl(std::shared_ptr<CombinedParams> params)
    : params(params), num_layers(params->net_params.num_input_layers)
{
    // 这里的cache是在PointRender.h中定义，等于说整体的网络定义就是在这里.
    cache = std::make_shared<PointRendererCache>();
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> PointRenderModuleImpl::forward(
    NeuralScene& scene, const std::vector<NeuralTrainData>& batch, CUDA::CudaTimerSystem* timer_system)
{
    NeuralRenderInfo render_data;
    render_data.scene        = &scene;
    render_data.num_layers   = num_layers;
    render_data.params       = params->render_params;
    render_data.timer_system = timer_system;

    if (!this->is_training())
    {
        render_data.params.dropout = 0;
    }

    for (auto& b : batch)
    {
        render_data.images.push_back(b->img);
    }

    return forward(&render_data);
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> PointRenderModuleImpl::forward(NeuralRenderInfo* nri)
{
    // 这里应该调试的时候用到的，现在不需要
    if (0)
    {
        auto poses = nri->scene->poses->Download();
        auto ks    = nri->scene->intrinsics->DownloadK();
        for (auto i : nri->images)
        {
            std::cout << "Render (" << i.camera_index << ", " << i.image_index << ") Pose: " << poses[i.image_index]
                      << " K: " << ks[i.camera_index] << std::endl;
        }
    }

    /**************************************************
     * 这里就是NeuralRenderInfo信息->等于说cache实例化成nri中的model信息
     **************************************************/
    nri->cache = cache.get();


    //这里是对图像尺度放大->应该是上采样
    if (params->render_params.super_sampling)
    {
        for (auto& i : nri->images)
        {
            i.w *= 2;
            i.h *= 2;
            i.crop_transform = i.crop_transform.scale(2);
        }
    }
    // 这里的Blend不知道是做啥的
    auto combined_images_masks = BlendPointCloud(nri);


    //这里才是具体的，nri当中的所有图像整合在一起才是渲染出来的图像
    std::vector<torch::Tensor> images(combined_images_masks.begin(), combined_images_masks.begin() + nri->num_layers);
    std::vector<torch::Tensor> point_masks;

    if (params->render_params.output_background_mask)
    {
        SAIGA_ASSERT(combined_images_masks.size() == nri->num_layers * 2);
        point_masks =
            std::vector<torch::Tensor>(combined_images_masks.begin() + nri->num_layers, combined_images_masks.end());

        for (auto& m : point_masks)
        {
            m.detach_();
        }
        SAIGA_ASSERT(!point_masks.front().requires_grad());
    }

    if (params->render_params.super_sampling)
    {
        // 这里论文有提，是将avg_pool来做上下采样中的像素填空？
        for (auto& img : images)
        {
            img = torch::avg_pool2d(img, {2, 2});
        }

        for (auto& img : point_masks)
        {
            img = torch::avg_pool2d(img, {2, 2});
        }

        for (auto& i : nri->images)
        {
            i.w /= 2;
            i.h /= 2;
            i.crop_transform = i.crop_transform.scale(0.5);
        }
    }



    // 这个是场景中构建scene中的environmental map数据结构
    if (nri->scene->environment_map)
    {
        SAIGA_ASSERT(params->render_params.output_background_mask);
        // 感觉这里是渲染出图像？
        auto env_maps = nri->scene->environment_map->Sample(
            nri->scene->poses->poses_se3, nri->scene->intrinsics->intrinsics, nri->images, nri->num_layers);

        if (params->pipeline_params.cat_env_to_color)
        {
            for (int i = 0; i < nri->num_layers; ++i)
            {
                images[i] = torch::cat({images[i], env_maps[i]}, 1);
            }
        }
        else
        {
            for (int i = 0; i < nri->num_layers; ++i)
            {
                auto background_mask = 1 - point_masks[i];
                images[i]            = images[i] + background_mask * env_maps[i];
            }
        }
    }

    if (params->pipeline_params.cat_masks_to_color)
    {
        for (int i = 0; i < nri->num_layers; ++i)
        {
            images[i] = torch::cat({images[i], point_masks[i]}, 1);
        }
    }

    SAIGA_ASSERT(images.front().size(0) == nri->images.size());
    SAIGA_ASSERT(images.front().size(2) == nri->images.front().h);
    SAIGA_ASSERT(images.front().size(3) == nri->images.front().w);



    return {images, point_masks};
}
