using System;
using System.Collections.Generic;
using System.Linq;
using Grasshopper.Kernel;
using MatchingWrapper.Properties;
using Rhino.Geometry;

namespace MatchingWrapper
{
    public class MinimumBoundindBoxComponent : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the MinimumBoundindBoxComponent class.
        /// </summary>
        public MinimumBoundindBoxComponent()
          : base("MinimumBoundindBox", "minBB",
              "Calculates the minimum Bounding box for a collection of points in cases when the plane is unknown.",
              "Surface", "Primitive")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGeometryParameter("Geometry", "geom", "Collection of Geometry to find minimum bounding box for.", GH_ParamAccess.list); // 0 
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddBoxParameter("Bounding Box", "bb", "Minimum bounding box", GH_ParamAccess.item); // 0    
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object is used to retrieve from inputs and store in outputs.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            List<Point3d> points = new List<Point3d>(); // 0

            List<object> objects = new List<object>(); // 0
            if (!DA.GetDataList(0, objects)) return;

            foreach (object obj in objects) 
            {
                
                var type = obj.GetType();

                Point3d point = new Point3d(); 
                Brep brep = new Brep();
                Mesh mesh = new Mesh(); 
                // Try cast to point
                if (GH_Convert.ToPoint3d(obj, ref point, GH_Conversion.Primary))
                {
                    points.Add(point);
                }
                // Try cast to Brep
                if (GH_Convert.ToBrep(obj, ref brep, GH_Conversion.Primary))
                {
                    points.AddRange(PointsFromBrep(brep));
                }
                // Try cast to mesh
                if(GH_Convert.ToMesh(obj, ref mesh, GH_Conversion.Primary))
                {
                    points.AddRange(PointsFromMesh(mesh));
                }
            }

            // Calculate the bounding box
            Box boundingBox = MinimumBoundindBox.GetMinimumBoundindBox(this, points); // Calculate the minimum bounding box

            // return the bounding box
            DA.SetData(0, boundingBox); // 0
        }
        
        private List<Point3d> PointsFromBrep(Brep brep)
        {
            return brep.DuplicateVertices().ToList();
        }

        private List<Point3d> PointsFromMesh(Mesh mesh)
        {
            return mesh.Vertices.ToPoint3dArray().ToList();
        }
        private List<Point3d> PointsFromSurface(Surface srf)
        {
            
            return srf.ToBrep().DuplicateVertices().ToList();
        }
        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                //You can add image files to your project resources and access them like this:
                // return Resources.IconForThisComponent;
                return Resources.min_bb;
            }
        }

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("08495E60-E97B-4AE7-944C-501A4407EDE8"); }
        }
    }
}