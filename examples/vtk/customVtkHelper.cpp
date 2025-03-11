/*
 * Author:
 * 2025/03/10 - Yuqing Liang (BIMCoder Liang)
 * bim.frankliang@foxmail.com
 *
 * Use of this source code is governed by a GPL-3.0 license that can be found in
 * the LICENSE file.
 */

#include "customVtkHelper.h"
#include "XYZ.h"
#include "XYZW.h"
#include "LNObject.h"
#include "Constants.h"
#include "NurbsCurve.h"

#include <vtkNew.h>
#include <vtkPolyData.h>
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkAutoInit.h>

#include <random>
VTK_MODULE_INIT(vtkRenderingOpenGL2)

void DisplayTargetCurve(vtkSmartPointer<vtkRenderer> renderer, const std::vector<XYZ>& tessellation)
{
	int size = tessellation.size();

	vtkNew<vtkPolyData> geometry;
	vtkNew<vtkPoints> points;
	vtkNew<vtkCellArray> polys;
	polys->InsertNextCell(size);
	for (int i = 0; i < size; i++)
	{
		LNLib::XYZ t = tessellation[i];
		points->InsertPoint(i, t.GetX(), t.GetY(), t.GetZ());
		polys->InsertCellPoint(i);
	}

	geometry->SetPoints(points);
	geometry->SetLines(polys);

	vtkNew<vtkPolyDataMapper> geometryMapper;
	geometryMapper->SetInputData(geometry);
	geometryMapper->ScalarVisibilityOff();
	vtkNew<vtkActor> geometryActor;

	double r = 1.0;
	double g = 0.0;
	double b = 0.0;

	geometryActor->GetProperty()->SetColor(r, g, b);
	geometryActor->GetProperty()->SetLineWidth(3);
	geometryActor->SetMapper(geometryMapper);

	renderer->AddActor(geometryActor);
}

void DisplayNurbsCurve(vtkSmartPointer<vtkRenderer> renderer, const LN_NurbsCurve& nurbsCurve3d)
{
	std::vector<LNLib::XYZ> tessellation = LNLib::NurbsCurve::Tessellate(nurbsCurve3d);
	int size = tessellation.size();

	vtkNew<vtkPolyData> geometry;
	vtkNew<vtkPoints> points;
	vtkNew<vtkCellArray> polys;
	polys->InsertNextCell(size);
	for (int i = 0; i < size; i++)
	{
		LNLib::XYZ t = tessellation[i];
		points->InsertPoint(i, t.GetX(), t.GetY(), t.GetZ());
		polys->InsertCellPoint(i);
	}

	geometry->SetPoints(points);
	geometry->SetLines(polys);

	vtkNew<vtkPolyDataMapper> geometryMapper;
	geometryMapper->SetInputData(geometry);
	geometryMapper->ScalarVisibilityOff();
	vtkNew<vtkActor> geometryActor;

	geometryActor->GetProperty()->SetColor(1, 1, 1);
	geometryActor->GetProperty()->SetLineWidth(3);
	geometryActor->SetMapper(geometryMapper);

	renderer->AddActor(geometryActor);
}

void DisplayControlPoints(vtkSmartPointer<vtkRenderer> renderer, std::vector<vtkSmartPointer<vtkActor>>& actors, const std::vector<XYZ>& points)
{
	for (int i = 0; i < points.size(); i++)
	{
		XYZ point = points[i];
		vtkNew<vtkSphereSource> source;
		source->SetRadius(2);
		source->SetCenter(point.GetX(), point.GetY(), point.GetZ());
		source->SetPhiResolution(10);

		vtkNew<vtkPolyDataMapper> mapper;
		mapper->SetInputConnection(source->GetOutputPort());
		vtkNew<vtkActor> actor;
		actor->SetMapper(mapper);
		actors.emplace_back(actor);

		renderer->AddActor(actor);
	}
}
