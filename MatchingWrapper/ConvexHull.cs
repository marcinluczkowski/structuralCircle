using System;
using Rhino.Geometry;
using Grasshopper;
using MIConvexHull;
using System.Collections.Generic;
using System.Xml.Schema;
using Grasshopper.Kernel.Geometry.Delaunay;
using System.Linq;
using Rhino.Geometry.Collections;
using System.Runtime.CompilerServices;
using Grasshopper.Kernel;

public class MinimumBoundindBox
{
	public static Box GetMinimumBoundindBox(GH_Component component, List<Point3d> inputPts)
	{
		// Use convex hull to find convex mesh of all points

		Mesh inputMesh = GetConvexHull3D(component, inputPts);

		if (!inputMesh.IsValid) { 
			component.AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Invalid mesh. Check that input is correct. Are all points planar?");
			return Box.Empty;
		}

        MeshFaceList faces = inputMesh.Faces; // Get all the faces of the mesh
											  // 
        List<Plane> planeList = new List<Plane>(); // list of the planes created by the three points in each face
		var points = inputMesh.TopologyVertices;
		foreach (MeshFace face in faces)
		{
			Point3f pt0 = points[face.A];
            Point3f pt1 = points[face.B];
            Point3f pt2 = points[face.C];

			Plane plane = new Plane(pt0, pt1, pt2); // create a plane
			if (plane != null)
			{
				planeList.Add(plane);
			}
        }
        
        List<Box> source = new List<Box>();
		// calculate the bounding box for all planes
        foreach (Plane plane in planeList)
        {
            Box box;
            ((GeometryBase)inputMesh).GetBoundingBox(plane, out box);
            source.Add(box);
        }
		// sort the bounding boxes by volume and return the smallest one
        return ((IEnumerable<Box>)((IEnumerable<Box>)source).OrderBy((Func<Box, double>)(o => ((Box) o).Volume))).ToList<Box>()[0];
    }


	/// <summary>
	/// Creates a ConvexHull mesh based on the input points
	/// </summary>
	/// <param name="inputPts"></param>
	/// <returns></returns>
	public class MIVertex : IVertex
	{
		public double[] Position { get; set; }
        public MIVertex(Point3d rhinoPt)
        {
            this.Position = new double[3]{rhinoPt.X, rhinoPt.Y, rhinoPt.Z};
        }

		public Point3d AsPoint3d()
		{
			return new Point3d(this.Position[0], this.Position[1], this.Position[2]);
		}
		public string AsString()
		{
			return String.Format("{0},{1},{2}", Position[0], Position[1], Position[2] );
		}
    }

	public class MIFace : ConvexFace<MIVertex, MIFace>
	{

	}

	private static Mesh GetConvexHull3D(GH_Component component, List<Point3d> inputPts)
	{
		Mesh convexHullMesh = new Mesh(); // instantiate the empty mesh
		List<MIVertex> vertexList = new List<MIVertex>(); // Instantiate empty list of MIVertex class
		foreach (Point3d point in inputPts)
		{
			vertexList.Add(new MIVertex(point)); // convert the Rhino geometry to MIVertex class
			
		}
		
		ConvexHullCreationResult<MIVertex, MIFace> hullResult = ConvexHull.Create<MIVertex, MIFace>(vertexList); // create the convex hull
		if (hullResult.Result != null)
		{
			List<MIVertex> resultVertices = hullResult.Result.Points.ToList(); // Get list of vertices as result
			List<MIFace> resultFaces = hullResult.Result.Faces.ToList(); // Get list of faces as result

			List<string> stringList1 = new List<string>(); // List of strings

			// Iterate through all vertices
			List<string> allVerticesString = new List<string>();
			foreach (MIVertex vertex in resultVertices)
			{

				//MeshVertexList vertices = convexHullMesh.Vertices;
				convexHullMesh.Vertices.Add(vertex.AsPoint3d());
				//vertices.Add(vertex.AsPoint3d()); // Add the vertex to the MeshVertexList		
				allVerticesString.Add(vertex.AsString()); //Add the vertex to all faces

			}

			// Iterate through all faces
			foreach (MIFace face in resultFaces)
			{
				MIVertex[] faceVertices = face.Vertices;

				int num0 = allVerticesString.IndexOf(faceVertices[0].AsString());
				int num1 = allVerticesString.IndexOf(faceVertices[1].AsString());
				int num2 = allVerticesString.IndexOf(faceVertices[2].AsString());

				convexHullMesh.Faces.AddFace(num0, num1, num2);


			}

			convexHullMesh.Normals.ComputeNormals();
			convexHullMesh.Compact();
			return convexHullMesh;
		}
		else
		{			
			component.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, hullResult.ErrorMessage);
			return convexHullMesh;
        }
	}
}
