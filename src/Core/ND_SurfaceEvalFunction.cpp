/*
 * Author:
 * Copyright (c) 2025/02/24- Yuqing Liang (BIMCoder Liang)
 * bim.frankliang@foxmail.com
 *
 * Use of this source code is governed by a GPL-3.0 license that can be found in
 * the LICENSE file.
 *
 */

#include "ND_SurfaceEvalFunction.h"

torch::Tensor ND_LNLib::SurfaceEvalFunction::forward(AutogradContext* ctx, torch::Tensor controlPoints, torch::Tensor uspan, torch::Tensor vspan, torch::Tensor uBasisFunctions, torch::Tensor vBasisFunctions, torch::Tensor uParamList, torch::Tensor vParamList, int uControlPointsCount, int vControlPointsCount, int degreeU, int degreeV, int dimension)
{
	ctx->save_for_backward({ controlPoints });
	ctx->saved_data["uspan"] = uspan;
	ctx->saved_data["vspan"] = vspan;
	ctx->saved_data["Nu"] = uBasisFunctions;
	ctx->saved_data["Nv"] = vBasisFunctions;
	ctx->saved_data["u"] = uParamList;
	ctx->saved_data["v"] = vParamList;
	ctx->saved_data["m"] = uControlPointsCount;
	ctx->saved_data["n"] = vControlPointsCount;
	ctx->saved_data["p"] = degreeU;
	ctx->saved_data["q"] = degreeV;
	ctx->saved_data["_dimension"] = dimension;

	torch::Tensor surface = torch::zeros({ controlPoints.size(0), uParamList.size(0), vParamList.size(0), dimension + 1 }, torch::requires_grad());
    for (int k = 0; k < controlPoints.size(0); k++)
    {
        for (int j = 0; j < vParamList.size(0); j++)
        {
            for (int i = 0; i < uParamList.size(0); i++)
            {
                auto temp = torch::zeros({ degreeV + 1, dimension + 1 });
                auto Sw = torch::zeros({ dimension + 1 });
                for (int l = 0; l <= degreeV; l++)
                {
                    for (int r = 0; r <= degreeU; r++)
                    {
                        temp[l] = temp[l] + uBasisFunctions[i][r].item<double>() * controlPoints[k][uspan[i].item<int>() - degreeU + r][vspan[j].item<int>() - degreeV + l];
                    }
                    Sw += vBasisFunctions[j][l].item<double>() * temp[l];
                }
                surface[k][i][j] = Sw;
            }
        }
    }
	
    ctx->saved_data["surface"] = surface;
    torch::Tensor result = surface.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, dimension) })
                                / surface.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), dimension }).unsqueeze(-1);
    return result;
}

tensor_list ND_LNLib::SurfaceEvalFunction::backward(AutogradContext* ctx, tensor_list grad_outputs)
{
    // 获取保存的变量
    torch::Tensor controlPoints = ctx->get_saved_variables()[0];
    torch::Tensor uspan = ctx->saved_data["uspan"].toTensor();
    torch::Tensor vspan = ctx->saved_data["vspan"].toTensor();
    torch::Tensor uBasisFunctions = ctx->saved_data["Nu"].toTensor();
    torch::Tensor vBasisFunctions = ctx->saved_data["Nv"].toTensor();
    torch::Tensor uParamList = ctx->saved_data["u"].toTensor();
    torch::Tensor vParamList = ctx->saved_data["v"].toTensor();
    int uControlPointsCount = ctx->saved_data["m"].toInt();
    int vControlPointsCount = ctx->saved_data["n"].toInt();
    int degreeU = ctx->saved_data["p"].toInt();
    int degreeV = ctx->saved_data["q"].toInt();
    int dimension = ctx->saved_data["_dimension"].toInt();
    torch::Tensor surface = ctx->saved_data["surface"].toTensor();

    // 在 backward 函数中获取 uParamList 和 vParamList 的尺寸
    int num_points_u = uParamList.size(0);
    int num_points_v = vParamList.size(0);

    torch::Tensor grad_output_3d = grad_outputs[0].view({
        grad_outputs[0].size(0), 
        num_points_u, 
        num_points_v, 
        dimension
    });
    
    // torch::Tensor grad_sw = torch::zeros({ grad_outputs[0].size(0), grad_outputs[0].size(1),grad_outputs[0].size(2), dimension + 1 }, torch::kDouble);
    torch::Tensor grad_sw = torch::zeros({grad_output_3d.size(0), num_points_u, num_points_v,dimension}, torch::kDouble);
    for (int i = 0; i < dimension; ++i) 
    {
        // grad_sw.slice(3, i, i + 1) = grad_outputs[i].unsqueeze(-1);
        grad_sw.slice(3, i, i + 1) = grad_output_3d.slice(3, i, i + 1);
    }
    for (int d = 0; d < dimension; ++d) 
    {    
        // torch::Tensor grad_sw_slice = grad_sw.slice(3, dimension, dimension + 1);
        // torch::Tensor grad_output_d = grad_outputs[d];
        // torch::Tensor surfaces_slice = surface.slice(3, dimension, dimension + 1);
        // grad_sw_slice += grad_output_d.div(surfaces_slice);
        torch::Tensor surfaces_slice = surface.slice(3, d, d + 1);
        torch::Tensor grad_output_d = grad_output_3d.slice(3, d, d + 1);
        grad_sw.slice(3, d, d + 1) += grad_output_d.div(surfaces_slice);
    }

    torch::Tensor grad_ctrl_pts = torch::zeros_like(controlPoints);

    // for (int k = 0; k < grad_outputs[0].size(0); k++)
    // {
    //     for (int j = 0; j < vParamList.size(0); j++)
    //     {
    //         for (int i = 0; i < uParamList.size(0); i++)
    //         {
    //             auto grad_temp = torch::zeros({ degreeV + 1, dimension + 1 });
    //             for (int l = 0; l <= degreeV; l++)
    //             {
    //                 grad_temp[l] = vBasisFunctions[j][l] * grad_outputs[k][i][j];
    //                 for (int r = 0; r <= degreeU; r++)
    //                 {
    //                     grad_ctrl_pts[k][uspan[i].item<int>() - degreeU + r][vspan[j].item<int>() - degreeV + l] = grad_ctrl_pts[k][uspan[i].item<int>() - degreeU + r][vspan[j].item<int>() - degreeV + l] + uBasisFunctions[i][r].item<double>() * grad_temp[l];
    //                 }
    //             }
    //         }
    //     }
    // }

    int batch_size = grad_output_3d.size(0);
    for (int k = 0; k < batch_size; k++) {
        for (int j = 0; j < num_points_v; j++) {
            for (int i = 0; i < num_points_u; i++) {
                for (int d = 0; d < dimension; ++d) {
                    auto grad_temp = torch::zeros({degreeV + 1});
                    for (int l = 0; l <= degreeV; l++) {
                        grad_temp[l] = vBasisFunctions[j][l] * grad_output_3d[k][i][j][d];
                        for (int r = 0; r <= degreeU; r++) {
                            grad_ctrl_pts[k][uspan[i].item<int>() - degreeU + r][vspan[j].item<int>() - degreeV + l] += 
                                uBasisFunctions[i][r].item<double>() * grad_temp[l];
                        }
                    }
                }
            }
        }
    }

    // return { grad_ctrl_pts[0], torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    return { grad_ctrl_pts, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
}