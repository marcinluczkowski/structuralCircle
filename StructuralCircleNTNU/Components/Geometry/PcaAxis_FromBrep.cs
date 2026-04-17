using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Geometry
{
    /// <summary>
    /// First principal axis from area-weighted random surface samples on the Brep (same pipeline as Element from Brep).
    /// </summary>
    public class PcaAxis_FromBrep : GH_Component
    {
        public PcaAxis_FromBrep()
            : base("PCA Axis from Brep", "PcaAxis",
                   "Dominant axis from PCA on surface sample points (identical to Element from Brep).",
                   "StructuralCircleNTNU", "Geometry")
        { }

        public override Guid ComponentGuid => new Guid("4b91e700-2a8f-4d1c-b5e3-7c6d5e4f3a12");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBrepParameter("Brep", "B", "Brep to analyse.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddLineParameter ("Axis",         "Ax", "PCA axis line (OBB extent along first principal direction).", GH_ParamAccess.item);
            pManager.AddPointParameter("SamplePoints", "Pts", "Surface sample points used for PCA (same as internal algorithm).", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Brep brep = null;
            if (!DA.GetData(0, ref brep)) return;

            if (!BrepElementBuilder.TryGetPcaAxis(brep, out Line axis, out string message, out List<Point3d> samples))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, message ?? "PCA failed.");
                if (samples != null && samples.Count > 0)
                    DA.SetDataList(1, samples);
                return;
            }

            DA.SetData(0, axis);
            DA.SetDataList(1, samples);
        }
    }
}
