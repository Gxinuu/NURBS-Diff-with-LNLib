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
#include <vector>

void CurveFitting()
{
	int num_ctrl_pts = 32;
	int degree = 3;
	int num_eval_pts = 128;

	ND_LNLib::CurveEvalModule cm(num_ctrl_pts, degree, num_eval_pts);

	
	std::vector<torch::optim::OptimizerParamGroup> group;
	torch::optim::AdamOptions options(0.1);
	torch::optim::Adam opt(group, options);
	torch::optim::ReduceLROnPlateauScheduler rp();

	for (int i = 0; i < 100000; i++)
	{
		opt.zero_grad();

	}
}

int main(int, char* [])
{
	
}
