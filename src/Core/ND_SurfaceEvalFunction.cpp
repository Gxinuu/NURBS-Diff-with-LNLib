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
	return torch::Tensor();
}

tensor_list ND_LNLib::SurfaceEvalFunction::backward(AutogradContext* ctx, tensor_list grad_outputs)
{
	return tensor_list();
}
