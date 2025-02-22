/*
 * Author:
 * Copyright (c) 2025/02/04- Yuqing Liang (BIMCoder Liang)
 * bim.frankliang@foxmail.com
 *
 * Use of this source code is governed by a GPL-3.0 license that can be found in
 * the LICENSE file.
 *
 */

#include "ND_CurveEvalFunction.h"

torch::Tensor ND_LNLib::CurveEvalFunction::forward(AutogradContext* ctx, torch::Tensor controlPoints, torch::Tensor uspan, torch::Tensor basisFunctions, torch::Tensor paramList, int degree, int dimension)
{
	ctx->save_for_backward({ controlPoints });
	ctx->saved_data["uspan"] = uspan;
	ctx->saved_data["Nu"] = basisFunctions;
	ctx->saved_data["u"] = paramList;
	ctx->saved_data["p"] = degree;
	ctx->saved_data["_dimension"] = dimension;
	
	torch::Tensor curve = torch::zeros({ controlPoints.size(0), paramList.size(0), dimension + 1 }, torch::requires_grad());
	for (int k = 0; k < controlPoints.size(0); k++)
	{
		for (int x = 0; x < paramList.size(0); x++)
		{
			torch::Tensor Cw = torch::zeros({ dimension + 1 });
			// similar with LNLib NurbsCurve::GetPointOnCurve()
			for (int i = 0; i <= degree; i++)
			{
				Cw = Cw + basisFunctions[x][i].item<double>() * controlPoints[k][uspan[x].item<int>() - degree + i];
			}
			curve[k][x] = Cw;
		}
	}
	ctx->saved_data["curve"] = curve;

	torch::Tensor numerator = curve.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, dimension) });
	torch::Tensor denominator = curve.index({ torch::indexing::Slice(), torch::indexing::Slice(), dimension }).unsqueeze(-1);
	torch::Tensor result = numerator / denominator;
	return result;
}

tensor_list ND_LNLib::CurveEvalFunction::backward(AutogradContext* ctx, tensor_list grad_outputs)
{
	torch::Tensor controlPoints = ctx->get_saved_variables()[0];
	torch::Tensor uspan = ctx->saved_data["uspan"].toTensor();
	torch::Tensor basisFunctions = ctx->saved_data["Nu"].toTensor();
	torch::Tensor paramList = ctx->saved_data["u"].toTensor();
	int degree = ctx->saved_data["p"].toInt();
	int dimension = ctx->saved_data["_dimension"].toInt();
	torch::Tensor curve = ctx->saved_data["curve"].toTensor();

	torch::Tensor grad_cw = torch::zeros({ grad_outputs[0].size(0), grad_outputs[0].size(1), dimension + 1 }, torch::kDouble);
	
	grad_cw.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, dimension) }, grad_outputs[0]);
	for (int d = 0; d < dimension; ++d) {
		grad_cw.index({ torch::indexing::Slice(), torch::indexing::Slice(), dimension }) +=
			grad_outputs[0].index({ torch::indexing::Slice(), torch::indexing::Slice(), d }) /
			curve.index({ torch::indexing::Slice(), torch::indexing::Slice(), dimension });
	}

	torch::Tensor grad_ctrl_pts = torch::zeros_like(controlPoints);
	for (int k = 0; k < grad_cw.size(0); k++)
	{
		for (int i = 0; i < paramList.size(0); i++)
		{
			torch::Tensor temp = grad_cw[k];
			torch::Tensor grad_ctrl_pts_i = torch::zeros_like(controlPoints[k]);
			for (int j = 0; j <= degree; j++)
			{
				grad_ctrl_pts_i[uspan[i].item<int>() - degree + j] = grad_ctrl_pts_i[uspan[i].item<int>() - degree + j] + basisFunctions[i][j].item<float>() * temp[i];
			}
			grad_ctrl_pts[k] = (grad_ctrl_pts[k] + grad_ctrl_pts_i);
		}
	}

	return { grad_ctrl_pts, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor() };
}
