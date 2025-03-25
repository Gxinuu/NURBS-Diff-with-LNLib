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

	for (int i = 0; i <= _degreeU; i++) {
		knotVectorU[i] = 0.0;
		knotVectorU[knotVectorU.size() - 1 - i] = 1.0;
	}

	int internal = _uControlPointsCount - _degreeU - 1;
	if (internal > 0) {
		const double spacing = 1.0 / (internal + 1); 
		for (int j = 0; j < internal; ++j) {
			const int index = _degreeU + 1 + j;
			knotVectorU[index] = (j + 1) * spacing;
		}
	}

	_knotVectorU = torch::from_blob(knotVectorU.data(), { static_cast<long>(knotVectorU.size()) }, torch::kDouble);
	_uParamList = torch::linspace(knotVectorU[0], knotVectorU[knotVectorU.size() - 1], _uEvalCount, torch::kDouble);

	int n = (_vControlPointsCount - 1) + _degreeV + 1;
	knotVectorCount = n + 1;
	std::vector<double> knotVectorV(knotVectorCount);

	for (int i = 0; i <= _degreeV; i++) {
		knotVectorV[i] = 0.0;
		knotVectorV[knotVectorV.size() - 1 - i] = 1.0;
	}

	internal = _vControlPointsCount - _degreeV - 1;
	if (internal > 0) {
		const double spacing = 1.0 / (internal + 1); 

		for (int j = 0; j < internal; ++j) {
			const int index = _degreeV + 1 + j;
			knotVectorV[index] = (j + 1) * spacing;
		}
	}

	_knotVectorV = torch::from_blob(knotVectorV.data(), { static_cast<long>(knotVectorV.size()) }, torch::kDouble);
	_vParamList = torch::linspace(knotVectorV[0], knotVectorV[knotVectorV.size() - 1], _vEvalCount, torch::kDouble);

	int paramSize = _uParamList.size(0);

	std::vector<int> uSpanList;
	uSpanList.reserve(paramSize);

	std::vector<torch::Tensor> Nu;
	Nu.reserve(paramSize);
	for (int i = 0; i < paramSize; i++)
	{
		std::vector<double> N(_degreeU + 1);
		double u_param = _uParamList[i].item<double>();
		int spanIndex = LNLib::Polynomials::GetKnotSpanIndex(_degreeU, knotVectorU, u_param);
		LNLib::Polynomials::BasisFunctions(spanIndex, _degreeU, knotVectorU, u_param, N.data());
		// 深拷贝数据到Tensor
		torch::Tensor Nu_Tensor = torch::tensor(N, torch::dtype(torch::kDouble).requires_grad(false));
		uSpanList.emplace_back(spanIndex);
		Nu.emplace_back(Nu_Tensor);
	}

	_uspan = torch::from_blob(uSpanList.data(), torch::IntList(paramSize), torch::TensorOptions().dtype(torch::kInt)).clone();
	_uBasisFunctions = torch::stack(Nu);

	paramSize = _vParamList.size(0);

	std::vector<int> vSpanList;
	vSpanList.reserve(paramSize);

	std::vector<torch::Tensor> Nv;
	Nv.reserve(paramSize);
	for (int i = 0; i < paramSize; i++)
	{
		std::vector<double> N(_degreeV + 1);
		double v_param = _vParamList[i].item<double>();
		int spanIndex = LNLib::Polynomials::GetKnotSpanIndex(_degreeV, knotVectorV, v_param);
		LNLib::Polynomials::BasisFunctions(spanIndex, _degreeV, knotVectorV, v_param, N.data());
		// 深拷贝数据到Tensor
		torch::Tensor Nv_Tensor = torch::tensor(N, torch::dtype(torch::kDouble).requires_grad(false));
		vSpanList.emplace_back(spanIndex);
		Nv.emplace_back(Nv_Tensor);
	}

	_vspan = torch::tensor(vSpanList, torch::kInt32).clone();
	_vBasisFunctions = torch::stack(Nv);

	_uBasisFunctions.view({ _uEvalCount, degreeU + 1 });
	_vBasisFunctions.view({ _vEvalCount, degreeV + 1 });
}

torch::Tensor ND_LNLib::SurfaceEvalModule::forward(torch::Tensor input)
{
	SurfaceEvalFunction function;
	return function.apply(input, _uspan, _vspan, _uBasisFunctions, _vBasisFunctions, _uParamList, _vParamList, _uControlPointsCount, _vControlPointsCount, _degreeU, _degreeV, _dimension);
}
