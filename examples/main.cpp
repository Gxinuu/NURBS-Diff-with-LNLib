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
#include "ND_SurfaceEvalModule.h"
#include "XYZ.h"
#include "XYZW.h"
#include "Constants.h"
#include "LNObject.h"

#include <vtkNew.h>
#include <vtkAxesActor.h>
#include <vtkActor.h>
#include <vtkTransform.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkAutoInit.h>
#include <vtkCallbackCommand.h>
#include <vtkRendererCollection.h>

#include "customIteractorStyle.h"
#include "customVtkHelper.h"

#include <vector>
#include <cmath>

using namespace LNLib;
VTK_MODULE_INIT(vtkRenderingOpenGL2)


void CurveFitting(vtkSmartPointer<vtkRenderer> renderer)
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

    DisplayTargetCurve(renderer, targetPoints);

    torch::Tensor target_tensor = torch::zeros({ 1, num_eval_pts, 3 });
    for (int i = 0; i < num_eval_pts; ++i) {
        target_tensor[0][i][0] = targetPoints[i].X();
        target_tensor[0][i][1] = targetPoints[i].Y();
        target_tensor[0][i][2] = targetPoints[i].Z();
    }

    // Sample from the target point, to initialize the control point
    std::vector<LNLib::XYZ> ctrl_points(num_ctrl_pts);
    double sample_step = (num_eval_pts - 1) / double(num_ctrl_pts - 1);
    for (int i = 0; i < num_ctrl_pts; ++i) {
        int idx = std::round(i * sample_step);
        ctrl_points[i] = targetPoints[std::min(idx, num_eval_pts - 1)];
    }

    // Create tensor
    torch::Tensor ctrl_pts_tensor = torch::zeros({ 1, num_ctrl_pts, 3 });
    for (int i = 0; i < num_ctrl_pts; ++i) {
        ctrl_pts_tensor[0][i][0] = ctrl_points[i].X();
        ctrl_pts_tensor[0][i][1] = ctrl_points[i].Y();
        ctrl_pts_tensor[0][i][2] = ctrl_points[i].Z();
    }
    ctrl_pts_tensor = ctrl_pts_tensor.requires_grad_(true); // CPU

    ND_LNLib::CurveEvalModule cm(num_ctrl_pts, degree, num_eval_pts);

    std::vector<torch::Tensor> group{ ctrl_pts_tensor };
    torch::optim::AdamOptions options(lr);
    torch::optim::Adam opt(group, options);
    torch::optim::ReduceLROnPlateauScheduler scheduler(opt, torch::optim::ReduceLROnPlateauScheduler::SchedulerMode::min);

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        opt.zero_grad();
        torch::Tensor weights = torch::ones({ 1, num_ctrl_pts, 1 }); // CPU
        torch::Tensor pred = cm.forward(torch::cat({ ctrl_pts_tensor, weights }, -1));

        auto dist = torch::cdist(pred, target_tensor);
        auto loss1 = torch::mean(torch::amin(dist, 2));
        auto loss2 = torch::mean(torch::amin(dist, 1));
        auto chamfer_loss = (loss1 + loss2) / 2.0;

        torch::Tensor total_loss = chamfer_loss;
        if (epoch < 3000) {
            auto diff = pred.slice(1, 0, -1) - pred.slice(1, 1);
            auto curve_length = torch::mean(torch::sum(diff.square(), { 1,2 }));
            total_loss += curve_length * 0.1;
        }

        total_loss.backward();
        opt.step();

        scheduler.step(total_loss.item<float>());

        if (epoch % 1000 == 0) {

            double curveLoss = total_loss.item<double>();

            // Save control points
            std::vector<XYZW> fittingControlPoints;
            auto ctrl_data = ctrl_pts_tensor.accessor<float, 3>();
            for (int i = 0; i < ctrl_data.size(1); ++i) {
                double x = ctrl_data[0][i][0];
                double y = ctrl_data[0][i][1];
                double z = ctrl_data[0][i][2];

                XYZW fp = XYZW(XYZ(x, y, z), 1);
                fittingControlPoints.emplace_back(fp);
            }

            int m = num_ctrl_pts + degree;
            std::vector<double> fittingKnotVector(m + 1);

            for (int i = 0; i <= m; ++i) {
                if (i <= degree) {
                    fittingKnotVector[i] = 0.0;
                }
                else if (i >= m - degree) {
                    fittingKnotVector[i] = 1.0;
                }
                else {
                    fittingKnotVector[i] = static_cast<double>(i - degree) / (num_ctrl_pts - degree);
                }
            }

            LN_NurbsCurve curve;
            curve.Degree = degree;
            curve.KnotVector = fittingKnotVector;
            curve.ControlPoints = fittingControlPoints;

            DisplayNurbsCurve(renderer, curve);

            // Save prediction points
            auto pred_data = pred.accessor<float, 3>();
            for (int i = 0; i < pred_data.size(1); ++i) {
                double x = pred_data[0][i][0];
                double y = pred_data[0][i][1];
                double z = pred_data[0][i][2];
            }
        }
    }    
}

bool _is_curve_fitting = true;

void SurfaceFitting(vtkSmartPointer<vtkRenderer> renderer)
{
    int degreeU = 3;
    int degreeV = 3;
    std::vector<double> kvU = { 0,0,0,0,0.4,0.6,1,1,1,1 };
    std::vector<double> kvV = { 0,0,0,0,0.4,0.6,1,1,1,1 };
    std::vector<std::vector<XYZW>> controlPoints(6, std::vector<XYZW>(6));

    controlPoints[0][0] = XYZW(0, 0, 0, 1);
    controlPoints[0][1] = XYZW(6.666666, 0, 4, 1);
    controlPoints[0][2] = XYZW(16.666666, 0, 22, 1);
    controlPoints[0][3] = XYZW(33.333333, 0, 22, 1);
    controlPoints[0][4] = XYZW(43.333333, 0, 4, 1);
    controlPoints[0][5] = XYZW(50, 0, 0, 1);

    controlPoints[1][0] = XYZW(0, 6.666667, 0, 1);
    controlPoints[1][1] = XYZW(6.6666667, 6.666667, 9.950068, 1);
    controlPoints[1][2] = XYZW(16.6666666, 6.666667, 9.65541838, 1);
    controlPoints[1][3] = XYZW(33.3333333, 6.666667, 47.21371742, 1);
    controlPoints[1][4] = XYZW(43.3333333, 6.666667, -11.56982167, 1);
    controlPoints[1][5] = XYZW(50, 6.6666667, 0, 1);

    controlPoints[2][0] = XYZW(0, 16.666666, 0, 1);
    controlPoints[2][1] = XYZW(6.6666667, 16.666666, -7.9001371, 1);
    controlPoints[2][2] = XYZW(16.6666666, 16.666666, 46.6891632, 1);
    controlPoints[2][3] = XYZW(33.3333333, 16.666667, -28.4274348, 1);
    controlPoints[2][4] = XYZW(43.3333333, 16.666667, 35.1396433, 1);
    controlPoints[2][5] = XYZW(50, 16.6666667, 0, 1);

    controlPoints[3][0] = XYZW(0, 33.3333333, 0, 1);
    controlPoints[3][1] = XYZW(6.6666667, 33.3333333, 29.2877911, 1);
    controlPoints[3][2] = XYZW(16.6666666, 33.3333333, -30.4644718, 1);
    controlPoints[3][3] = XYZW(33.3333333, 33.3333333, 129.1582990, 1);
    controlPoints[3][4] = XYZW(43.3333333, 33.3333333, -62.1717142, 1);
    controlPoints[3][5] = XYZW(50, 33.333333, 0, 1);

    controlPoints[4][0] = XYZW(0, 43.333333, 0, 1);
    controlPoints[4][1] = XYZW(6.6666667, 43.333333, -10.384636, 1);
    controlPoints[4][2] = XYZW(16.6666666, 43.333333, 59.21371742, 1);
    controlPoints[4][3] = XYZW(33.3333333, 43.333333, -37.7272976, 1);
    controlPoints[4][4] = XYZW(43.3333333, 43.333333, 45.1599451, 1);
    controlPoints[4][5] = XYZW(50, 43.333333, 0, 1);

    controlPoints[5][0] = XYZW(0, 50, 0, 1);
    controlPoints[5][1] = XYZW(6.6666667, 50, 0, 1);
    controlPoints[5][2] = XYZW(16.6666666, 50, 0, 1);
    controlPoints[5][3] = XYZW(33.3333333, 50, 0, 1);
    controlPoints[5][4] = XYZW(43.3333333, 50, 0, 1);
    controlPoints[5][5] = XYZW(50, 50, 0, 1);

    LN_NurbsSurface surface;
    surface.DegreeU = degreeU;
    surface.DegreeV = degreeV;
    surface.KnotVectorU = kvU;
    surface.KnotVectorV = kvV;
    surface.ControlPoints = controlPoints;
}

int main(int, char* [])
{
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetWindowName("ND_LNLib - BIMCoder Liang (bim.frankliang@foxmail.com)");
    renderWindow->FullScreenOn();
    renderWindow->BordersOn();

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    
    if (_is_curve_fitting)
    {
        CurveFitting(renderer);
       
    }
    else
    {
        SurfaceFitting(renderer);
    }

    vtkNew<vtkAxesActor> axesActor;
    vtkNew<vtkTransform>  userTrans;
    userTrans->Update();
    axesActor->SetUserTransform(userTrans);
    axesActor->AxisLabelsOn();
    axesActor->SetTotalLength(100, 100, 100);
    renderer->AddActor(axesActor);
    renderer->SetBackground(0, 0, 0);

    vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
    renderWindowInteractor->SetRenderWindow(renderWindow);

    vtkNew<customIteractorStyle> style;
    style->SetFixedAxesActor(axesActor);
    renderWindowInteractor->SetInteractorStyle(style);

    renderWindow->AddRenderer(renderer);
    renderer->ResetCamera();
    renderWindow->Render();

    renderWindowInteractor->Start();
    return 0;
}
