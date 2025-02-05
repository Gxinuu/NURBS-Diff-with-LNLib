/*
 * Author:
 * Copyright (c) 2025/02/24- Yuqing Liang (BIMCoder Liang)
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
	int m = (_controlPointsCount - 1) + _degree + 1;
	int knotVectorCount = m + 1;
	int segments = knotVectorCount - (2 * (_degree + 1)) + 1;
	double spacing = 1.0 / (double)segments;

	std::vector<double> knotVector(knotVectorCount);
	for (int i = 0; i <= degree; i++)
	{
		knotVector[i] = 0.0;
		knotVector[knotVector.size() - 1 - i] = 1.0;
	}

	for (int i = degree + 2; i <= knotVectorCount - 1 - degree; i++)
	{
		knotVector[i] = 0.0 + i * spacing;
	}

	_knotVector = torch::from_blob(knotVector.data(), { static_cast<long>(knotVector.size()) }, torch::kDouble);
	_paramList = torch::linspace(0.0, 1.0, _evalCount, torch::kDouble);

	int paramSize = _paramList.size(0);

	std::vector<int> spanList;
	spanList.reserve(paramSize);

	std::vector<torch::Tensor> Nu;
	Nu.reserve(paramSize);
	for (int i = 0; i < paramSize; i++)
	{
		int spanIndex = LNLib::Polynomials::GetKnotSpanIndex(_degree, knotVector, _paramList[i].item<double>());
		double N[LNLib::Constants::NURBSMaxDegree + 1];
		LNLib::Polynomials::BasisFunctions(spanIndex, _degree, knotVector, _paramList[i].item<double>(), N);
		spanList.emplace_back(spanIndex);
		torch::Tensor Nu_Tensor = torch::from_blob(N, { LNLib::Constants::NURBSMaxDegree + 1 }, torch::kDouble);
		Nu.emplace_back(Nu_Tensor);
	}

	_uspan = torch::from_blob(spanList.data(), torch::IntList(paramSize), torch::TensorOptions().dtype(torch::kInt));
	_basisFunctions = torch::stack(Nu);
}

torch::Tensor ND_LNLib::CurveEvalModule::forward(torch::Tensor x)
{
	CurveEvalFunction function;
	return function.apply(x, _uspan, _basisFunctions, _paramList, _degree, _dimension);
}
