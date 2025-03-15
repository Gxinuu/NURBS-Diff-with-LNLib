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
#include "NurbsSurface.h"

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
#include <vtkTriangle.h>

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

void DisplayNurbsCurve(vtkSmartPointer<vtkRenderer> renderer, const LN_NurbsCurve& nurbsCurve)
{
	std::vector<LNLib::XYZ> tessellation = LNLib::NurbsCurve::Tessellate(nurbsCurve);
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

void DisplaySurface(vtkSmartPointer<vtkRenderer> renderer, const LN_NurbsSurface& nurbsSurface, bool isTarget)
{
	LNLib::LN_Mesh mesh = NurbsSurface::Triangulate(nurbsSurface);
	std::vector<std::vector<int>> faces = mesh.Faces;
	std::vector<XYZ> vertices = mesh.Vertices;

	// Setup three points
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	for (int i = 0; i < vertices.size(); i++)
	{
		points->InsertNextPoint(vertices[i][0], vertices[i][1], vertices[i][2]);
	}

	// Add the polygon to a list of polygons
	vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();
	for (int i = 0; i < faces.size(); i++)
	{
		// Create the triangle
		vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();

		triangle->GetPointIds()->SetId(0, faces[i][0]);
		triangle->GetPointIds()->SetId(1, faces[i][1]);
		triangle->GetPointIds()->SetId(2, faces[i][2]);

		triangles->InsertNextCell(triangle);
	}

	// Create a PolyData
	vtkSmartPointer<vtkPolyData> polygonPolyData = vtkSmartPointer<vtkPolyData>::New();
	polygonPolyData->SetPoints(points);
	polygonPolyData->SetPolys(triangles);

	// Create a mapper and actor
	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(polygonPolyData);

	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	if (isTarget)
	{
		actor->GetProperty()->SetColor(1, 0, 0);
	}
	else
	{
		actor->GetProperty()->SetColor(1, 1, 1);
	}
	renderer->AddActor(actor);
}

