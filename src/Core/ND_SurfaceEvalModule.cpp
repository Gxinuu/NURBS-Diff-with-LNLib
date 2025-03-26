/*
 * Author:
 * Copyright (c) 2025/02/24- Yuqing Liang (BIMCoder Liang)
 * bim.frankliang@foxmail.com
 *
 * Use of this source code is governed by a GPL-3.0 license that can be found in
 * the LICENSE file.
 *
 */

#include "ND_SurfaceEvalModule.h"
#include "ND_SurfaceEvalFunction.h"
#include "Polynomials.h"
#include "Constants.h"
#include <vector>

ND_LNLib::SurfaceEvalModule::SurfaceEvalModule(int uControlPointsCount, int vControlPointsCount, int degreeU, int degreeV, int uEvalCount, int vEvalCount, int dimension) :
	_uControlPointsCount(uControlPointsCount), _vControlPointsCount(vControlPointsCount), _degreeU(degreeU), _degreeV(degreeV), _uEvalCount(uEvalCount), _vEvalCount(vEvalCount), _dimension(dimension)
{

	int m = (_uControlPointsCount - 1) + _degreeU + 1;
	int knotVectorCount = m + 1;
	std::vector<double> knotVectorU(knotVectorCount);

	for (int i = 0; i <= m; ++i) {
		if (i <= _degreeU) {
			knotVectorU[i] = 0.0;
		}
		else if (i >= m - _degreeU) {
			knotVectorU[i] = 1.0;
		}
		else {
			knotVectorU[i] = static_cast<double>(i - _degreeU) / (_uControlPointsCount - _degreeU);
		}
	}

	_knotVectorU = torch::from_blob(knotVectorU.data(), { static_cast<long>(knotVectorU.size()) }, torch::kDouble);
	_uParamList = torch::linspace(knotVectorU[0], knotVectorU[knotVectorU.size() - 1], _uEvalCount, torch::kDouble);

	int paramSize = _uParamList.size(0);

	std::vector<int> uSpanList;
	uSpanList.reserve(paramSize);

	std::vector<torch::Tensor> Nu;
	Nu.reserve(paramSize);
	for (int i = 0; i < paramSize; i++)
	{
		int spanIndex = LNLib::Polynomials::GetKnotSpanIndex(_degreeU, knotVectorU, _uParamList[i].item<double>());

		std::vector<double> N(_degreeU + 1);
		LNLib::Polynomials::BasisFunctions(spanIndex, _degreeU, knotVectorU, _uParamList[i].item<double>(), N.data());

		uSpanList.emplace_back(spanIndex);

		torch::Tensor Nu_Tensor = torch::tensor(
			torch::ArrayRef<double>(N.data(), _degreeU + 1),
			torch::kDouble
		);
		Nu.emplace_back(Nu_Tensor);
	}

	_uspan = torch::from_blob(uSpanList.data(), torch::IntList(paramSize), torch::TensorOptions().dtype(torch::kInt)).clone();
	_uBasisFunctions = torch::stack(Nu);


	int n = (_vControlPointsCount - 1) + _degreeV + 1;
	knotVectorCount = n + 1;
	std::vector<double> knotVectorV(knotVectorCount);

	for (int i = 0; i <= n; ++i) {
		if (i <= _degreeV) {
			knotVectorV[i] = 0.0;
		}
		else if (i >= n - _degreeV) {
			knotVectorV[i] = 1.0;
		}
		else {
			knotVectorV[i] = static_cast<double>(i - _degreeV) / (_vControlPointsCount - _degreeV);
		}
	}

	_knotVectorV = torch::from_blob(knotVectorV.data(), { static_cast<long>(knotVectorV.size()) }, torch::kDouble);
	_vParamList = torch::linspace(knotVectorV[0], knotVectorV[knotVectorV.size() - 1], _vEvalCount, torch::kDouble);

	paramSize = _vParamList.size(0);

	std::vector<int> vSpanList;
	vSpanList.reserve(paramSize);

	std::vector<torch::Tensor> Nv;
	Nv.reserve(paramSize);
	for (int i = 0; i < paramSize; i++)
	{
		int spanIndex = LNLib::Polynomials::GetKnotSpanIndex(_degreeV, knotVectorV, _vParamList[i].item<double>());

		std::vector<double> N(_degreeV + 1);
		LNLib::Polynomials::BasisFunctions(spanIndex, _degreeV, knotVectorV, _vParamList[i].item<double>(), N.data());

		vSpanList.emplace_back(spanIndex);

		torch::Tensor Nv_Tensor = torch::tensor(
			torch::ArrayRef<double>(N.data(), _degreeV + 1),
			torch::kDouble
		);
		Nv.emplace_back(Nv_Tensor);
	}

	_vspan = torch::from_blob(vSpanList.data(), torch::IntList(paramSize), torch::TensorOptions().dtype(torch::kInt)).clone();
	_vBasisFunctions = torch::stack(Nv);

	_uBasisFunctions.view({ _uEvalCount, degreeU + 1 });
	_vBasisFunctions.view({ _vEvalCount, degreeV + 1 });
}

torch::Tensor ND_LNLib::SurfaceEvalModule::forward(torch::Tensor input)
{
	SurfaceEvalFunction function;
	return function.apply(input, _uspan, _vspan, _uBasisFunctions, _vBasisFunctions, _uParamList, _vParamList, _uControlPointsCount, _vControlPointsCount, _degreeU, _degreeV, _dimension);
}
