/*
 * Author:
 * Copyright (c) 2025/02/04- Yuqing Liang (BIMCoder Liang)
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
	class ND_LNLib_EXPORT CurveEvalFunction:public Function<CurveEvalFunction>
	{
	public:

		static torch::Tensor forward(AutogradContext* ctx, torch::Tensor controlPoints, torch::Tensor uspan, torch::Tensor basisFunctions, torch::Tensor paramList, int degree, int dimension = 3);

		static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
	};

}



