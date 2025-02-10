/*
 * Author:
 * Copyright (c) 2025/02/05- Yuqing Liang (BIMCoder Liang)
 * bim.frankliang@foxmail.com
 *
 * Use of this source code is governed by a GPL-3.0 license that can be found in
 * the LICENSE file.
 *
 */

#include "ND_CurveEvalModule.h"
#include "XYZ.h"
#include "XYZW.h"
#include "Constants.h"
#include <vector>
#include <cmath>

void CurveFitting()
{
	int num_ctrl_pts = 32;
	int degree = 3;
	int num_eval_pts = 128;

	std::vector<LNLib::XYZ> targetPoints(num_eval_pts);
	double pStep = 2 * LNLib::Constants::Pi / (double)(num_eval_pts - 1);
	for (int i = 0; i < num_eval_pts; i++)
	{
		double x = 0 + i * pStep;
		double y = std::sin(x) + 2 * std::sin(2 * x) + sin(4 * x);
		double z = 0;
		targetPoints[i] = LNLib::XYZ(x, y, z);
	}



	ND_LNLib::CurveEvalModule cm(num_ctrl_pts, degree, num_eval_pts);

	std::vector<torch::optim::OptimizerParamGroup> group;
	torch::optim::AdamOptions options(0.1);
	torch::optim::Adam opt(group, options);
	torch::optim::ReduceLROnPlateauScheduler scheduler(opt, torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::min);

	for (int i = 0; i < 100000; i++)
	{
		opt.zero_grad();
		//
		opt.step();
		//
	}
}

int main(int, char* [])
{
	
}
