using Grasshopper;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;

namespace StructuralCircle
{
    public class StructuralCircleComponent : GH_Component
    {
        /// <summary>
        /// Each implementation of GH_Component must provide a public 
        /// constructor without any arguments.
        /// Category represents the Tab in which the component will appear, 
        /// Subcategory the panel. If you use non-existing tab or panel names, 
        /// new tabs/panels will automatically be created.
        /// </summary>
        public StructuralCircleComponent()
          : base("StructuralCircle", "GetBreps",
            "Take breps and categorize them",
            "Take", "TakeBrep")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddBrepParameter("breps","bs","breps which should be consider in the algorithm",GH_ParamAccess.list);
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("outputs","os","information about algorithm",GH_ParamAccess.list);
            pManager.AddLineParameter("axis", "as", "axis of the objects", GH_ParamAccess.list);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object can be used to retrieve data from input parameters and 
        /// to store data in output parameters.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            List<Brep> bs = new List<Brep>();
            DA.GetDataList(0, bs);

            List<string> info = new List<string>();
            List<Line> lines = new List<Line>();
            info.Add("¨---algorithm starts---");

            for(int i=0; i<bs.Count; i++)
            {
                Brep b = bs[i];
                Dictionary<int, double> dicFaceBr = new Dictionary<int, double>();
                int jf = 0;
                Point3d cenb = AreaMassProperties.Compute(b).Centroid;
                foreach(BrepFace bf in b.Faces)
                {
                    Point3d cenbf = AreaMassProperties.Compute(bf).Centroid;

                    double dis = Math.Abs(cenb.DistanceTo(cenbf));
                    dicFaceBr.Add(jf, dis);
                    jf++;
                }
                var sortedDic = dicFaceBr.OrderBy(x => x.Value).ToList();
                sortedDic.Reverse();
                var k1 = sortedDic[0].Key;
                var k2 = sortedDic[1].Key;

                var p1 = AreaMassProperties.Compute(b.Faces[k1]).Centroid;
                var p2 = AreaMassProperties.Compute(b.Faces[k2]).Centroid;

                Line axis = new Line(p1,p2);
                lines.Add(axis);
            }



            DA.SetDataList(0, info);
            DA.SetDataList(1, lines);
        }

        /// <summary>
        /// Provides an Icon for every component that will be visible in the User Interface.
        /// Icons need to be 24x24 pixels.
        /// You can add image files to your project resources and access them like this:
        /// return Resources.IconForThisComponent;
        /// </summary>
        protected override System.Drawing.Bitmap Icon => null;

        /// <summary>
        /// Each component must have a unique Guid to identify it. 
        /// It is vital this Guid doesn't change otherwise old ghx files 
        /// that use the old ID will partially fail during loading.
        /// </summary>
        public override Guid ComponentGuid => new Guid("B4F2E6EC-5EF8-488C-B516-C42C1B198B84");
    }
}