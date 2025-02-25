/*
 * Author:
 * Copyright (c) 2025/02/24- Yuqing Liang (BIMCoder Liang)
 * bim.frankliang@foxmail.com
 *
 * Use of this source code is governed by a GPL-3.0 license that can be found in
 * the LICENSE file.
 *
 */

#pragma once

#include "ND_Definitions.h"
#include <torch/torch.h>

using namespace torch::autograd;

namespace ND_LNLib
{
	class ND_LNLib_EXPORT SurfaceEvalFunction:public Function<SurfaceEvalFunction>
	{
	public:

		static torch::Tensor forward(AutogradContext* ctx, torch::Tensor controlPoints, torch::Tensor uspan, torch::Tensor vspan, torch::Tensor uBasisFunctions, torch::Tensor vBasisFunctions, torch::Tensor uParamList, torch::Tensor vParamList, int uControlPointsCount, int vControlPointsCount, int degreeU, int degreeV, int dimension = 3);

		static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
		
	};

}



