/*
 * Author:
 * Copyright (c) 2025/02/04- Yuqing Liang (BIMCoder Liang)
 * bim.frankliang@foxmail.com
 *
 * Use of this source code is governed by a GPL-3.0 license that can be found in
 * the LICENSE file.
 *
 */

#include "ND_CurveEvalModule.h"
#include "ND_CurveEvalFunction.h"
#include "Polynomials.h"
#include "Constants.h"
#include <vector>

ND_LNLib::CurveEvalModule::CurveEvalModule(int controlPointsCount, int degree, int evalCount, int dimension):
	_controlPointsCount(controlPointsCount),_degree(degree),_evalCount(evalCount),_dimension(dimension)
{

    int m = _controlPointsCount + _degree; 
	std::vector<double> knotVector(m + 1); 

	for (int i = 0; i <= m; ++i) {
		if (i <= _degree) {
			knotVector[i] = 0.0;
		} else if (i >= m - _degree) {
			knotVector[i] = 1.0;
		} else {
			knotVector[i] = static_cast<double>(i - _degree) / (_controlPointsCount - _degree);
		}
	}

	_knotVector = torch::from_blob(knotVector.data(), { static_cast<long>(knotVector.size()) }, torch::kDouble);
	_paramList = torch::linspace(knotVector[0], knotVector[knotVector.size() - 1], _evalCount, torch::kDouble);

	int paramSize = _paramList.size(0);

	std::vector<int> spanList;
	spanList.reserve(paramSize);

	std::vector<torch::Tensor> Nu;
	Nu.reserve(paramSize);

	for (int i = 0; i < paramSize; i++) 
	{
		int spanIndex = LNLib::Polynomials::GetKnotSpanIndex(_degree, knotVector, _paramList[i].item<double>());
		
		std::vector<double> N(LNLib::Constants::NURBSMaxDegree + 1); 
		LNLib::Polynomials::BasisFunctions(spanIndex, _degree, knotVector, _paramList[i].item<double>(), N.data());
		
		spanList.emplace_back(spanIndex);
		
		torch::Tensor Nu_Tensor = torch::tensor(
			torch::ArrayRef<double>(N.data(), LNLib::Constants::NURBSMaxDegree + 1), 
			torch::kDouble
		);
		Nu.emplace_back(Nu_Tensor);
	}

	_uspan = torch::from_blob(spanList.data(), torch::IntList(paramSize), torch::TensorOptions().dtype(torch::kInt)).clone();
	_basisFunctions = torch::stack(Nu);
}

torch::Tensor ND_LNLib::CurveEvalModule::forward(torch::Tensor input)
{
	CurveEvalFunction function;
	return function.apply(input, _uspan, _basisFunctions, _paramList, _degree, _dimension);
}
