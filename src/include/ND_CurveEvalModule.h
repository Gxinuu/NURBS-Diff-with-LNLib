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
	class ND_LNLib_EXPORT CurveEvalModule:torch::nn::Module
	{
	public:
		
		CurveEvalModule(int controlPointsCount, int degree, int evalCount, int dimension = 3);

		torch::Tensor forward(torch::Tensor x);

	private:

		int _controlPointsCount = 0;
		int _degree = 0;
		int _evalCount = 0;
		int _dimension = 3;
		torch::Tensor _knotVector;
		torch::Tensor _basisFunctions;
		torch::Tensor _uspan;
		torch::Tensor _paramList;
	};
}



