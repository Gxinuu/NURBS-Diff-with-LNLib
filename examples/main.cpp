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
#include <fstream>

void CurveFitting()
{
	int num_ctrl_pts = 64;
	int degree = 3;
	int num_eval_pts = 256;
	int epochs = 10000;
    double lr = 0.1;

	std::vector<LNLib::XYZ> targetPoints(num_eval_pts);
	double pStep = 2 * LNLib::Constants::Pi / (double)(num_eval_pts - 1);
	for (int i = 0; i < num_eval_pts; i++)
	{
		double x = 0 + i * pStep;
		double y = std::sin(x) + 2 * std::sin(2 * x) + sin(4 * x);
		double z = 0;
		targetPoints[i] = LNLib::XYZ(x, y, z);
	}

    torch::Tensor target_tensor = torch::zeros({1, num_eval_pts, 3});
    for(int i = 0; i < num_eval_pts; ++i) {
        target_tensor[0][i][0] = targetPoints[i].X();
        target_tensor[0][i][1] = targetPoints[i].Y();
        target_tensor[0][i][2] = targetPoints[i].Z();
    }

    // Sample from the target point, to initialize the control point
    std::vector<LNLib::XYZ> ctrl_points(num_ctrl_pts);
    double sample_step = (num_eval_pts-1) / double(num_ctrl_pts-1);
    for(int i = 0; i < num_ctrl_pts; ++i) {
        int idx = std::round(i * sample_step);
        ctrl_points[i] = targetPoints[std::min(idx, num_eval_pts-1)];
    }

    // Create tensor
    torch::Tensor ctrl_pts_tensor = torch::zeros({1, num_ctrl_pts, 3});
    for(int i = 0; i < num_ctrl_pts; ++i) {
        ctrl_pts_tensor[0][i][0] = ctrl_points[i].X();
        ctrl_pts_tensor[0][i][1] = ctrl_points[i].Y();
        ctrl_pts_tensor[0][i][2] = ctrl_points[i].Z();
    }
    ctrl_pts_tensor = ctrl_pts_tensor.requires_grad_(true); // CPU

    ND_LNLib::CurveEvalModule cm(num_ctrl_pts, degree, num_eval_pts);

	// Configure optimizer
	// std::vector<torch::optim::OptimizerParamGroup> group;
	std::vector<torch::Tensor> group{ctrl_pts_tensor};
	torch::optim::AdamOptions options(lr);
	torch::optim::Adam opt(group, options);
	torch::optim::ReduceLROnPlateauScheduler scheduler(opt, torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::min);

	for (int epoch = 1; epoch <= epochs; epoch++)
	{
		opt.zero_grad();
        torch::Tensor weights = torch::ones({1, num_ctrl_pts, 1}); // CPU
        torch::Tensor pred = cm.forward(torch::cat({ctrl_pts_tensor, weights}, -1));

        auto dist = torch::cdist(pred, target_tensor);
        auto loss1 = torch::mean(torch::amin(dist, 2));
        auto loss2 = torch::mean(torch::amin(dist, 1));
        auto chamfer_loss = (loss1 + loss2) / 2.0;

        torch::Tensor total_loss = chamfer_loss;
        if(epoch < 3000) {
            auto diff = pred.slice(1, 0, -1) - pred.slice(1, 1);
            auto curve_length = torch::mean(torch::sum(diff.square(), {1,2}));
            total_loss += curve_length * 0.1;
        }

        total_loss.backward();
		opt.step();
		
		scheduler.step(total_loss.item<float>()); 

        //
        // if(epoch % 1000 == 0) {
        //     std::cout << "Epoch [" << epoch << "/" << epochs 
        //              << "] Loss: " << total_loss.item<double>() << std::endl;
            
        //     torch::save(ctrl_pts_tensor, "ctrl_pts.pt");
        //     torch::save(pred.detach(), "pred.pt");
        //     torch::save(target_tensor, "target.pt");
        // }

        if (epoch % 1000 == 0) {
            std::cout << "Epoch [" << epoch << "/" << epochs 
                     << "] Loss: " << total_loss.item<double>() << std::endl;
            // Save control points
            std::ofstream ctrl_file("ctrl_pts.txt");
            auto ctrl_data = ctrl_pts_tensor.accessor<float, 3>();
            for (int i = 0; i < ctrl_data.size(1); ++i) {
                ctrl_file << ctrl_data[0][i][0] << " " 
                        << ctrl_data[0][i][1] << " " 
                        << ctrl_data[0][i][2] << "\n";
            }
            ctrl_file.close();

            // Save prediction points
            std::ofstream pred_file("pred.txt");
            auto pred_data = pred.accessor<float, 3>();
            for (int i = 0; i < pred_data.size(1); ++i) {
                pred_file << pred_data[0][i][0] << " " 
                        << pred_data[0][i][1] << " " 
                        << pred_data[0][i][2] << "\n";
            }
            pred_file.close();

            // Save target point
            std::ofstream target_file("target.txt");
            auto target_data = target_tensor.accessor<float, 3>();
            for (int i = 0; i < target_data.size(1); ++i) {
                target_file << target_data[0][i][0] << " " 
                            << target_data[0][i][1] << " " 
                            << target_data[0][i][2] << "\n";
            }
            target_file.close();
        }
	}
}

int main(int, char* [])
{
	CurveFitting();
	std::cout << "train finished" << std::endl;
	getchar();
    return 0;
}
