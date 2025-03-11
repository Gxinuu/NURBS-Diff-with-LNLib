/*
 * Author:
 * 2025/03/10 - Yuqing Liang (BIMCoder Liang)
 * bim.frankliang@foxmail.com
 *
 * Use of this source code is governed by a GPL-3.0 license that can be found in
 * the LICENSE file.
 */

#include "LNObject.h"
#include <vtkRenderer.h>
#include <vector>

using namespace LNLib;

void DisplayTargetCurve(vtkSmartPointer<vtkRenderer> renderer, const std::vector<XYZ>& tessellation);

void DisplayNurbsCurve(vtkSmartPointer<vtkRenderer> renderer, const LN_NurbsCurve& nurbsCurve3d);

void DisplayControlPoints(vtkSmartPointer<vtkRenderer> renderer, std::vector<vtkSmartPointer<vtkActor>>& actors, const std::vector<XYZ>& points);