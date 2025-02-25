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
	class ND_LNLib_EXPORT SurfaceEvalModule:torch::nn::Module
	{
	public:

		SurfaceEvalModule(int uControlPointsCount, int vControlPointsCount, int degreeU, int degreeV, int uEvalCount, int vEvalCount, int dimension = 3);

		torch::Tensor forward(torch::Tensor input);
		
	private:

		int _uControlPointsCount = 0;
		int _vControlPointsCount = 0;
		int _degreeU = 0;
		int _degreeV = 0;
		int _uEvalCount = 0;
		int _vEvalCount = 0;
		int _dimension = 3;
		torch::Tensor _knotVectorU;
		torch::Tensor _knotVectorV;
		torch::Tensor _uBasisFunctions;
		torch::Tensor _vBasisFunctions;
		torch::Tensor _uspan;
		torch::Tensor _vspan;
		torch::Tensor _uParamList;
		torch::Tensor _vParamList;
	};
}



