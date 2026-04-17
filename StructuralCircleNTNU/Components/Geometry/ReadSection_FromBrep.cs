using System;
using Grasshopper.Kernel;
using Rhino;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Geometry
{
    /// <summary>
    /// Estimates member length and cross-section width/height via PCA axis, three perpendicular plane cuts (25/50/75%),
    /// and via the same min-volume oriented box used in Min Volume Box from Brep.
    /// </summary>
    public class ReadSection_FromBrep : GH_Component
    {
        public ReadSection_FromBrep()
            : base("Read Section from Brep", "ReadSect",
                   "Length along PCA axis; width/height from plane–Brep intersections (averaged, 50% outlier reject) " +
                   "and from min-volume bounding box edge lengths.",
                   "StructuralCircleNTNU", "Geometry")
        { }

        public override Guid ComponentGuid => new Guid("a8f3c21d-9b0e-4c7a-8d2f-1e4b5c6d7e8f");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBrepParameter("Brep", "B", "Brep to measure (uses same 500-point PCA sampling as Element from Brep).", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddLineParameter("Axis", "Ax", "PCA axis line.", GH_ParamAccess.item);
            pManager.AddNumberParameter("Length", "L", "Length along PCA axis (m).", GH_ParamAccess.item);
            pManager.AddNumberParameter("WidthPlanes", "Wp", "Average section width from plane cuts (m); larger in-plane span.", GH_ParamAccess.item);
            pManager.AddNumberParameter("HeightPlanes", "Hp", "Average section height from plane cuts (m); smaller in-plane span.", GH_ParamAccess.item);
            pManager.AddNumberParameter("LengthBox", "Lb", "Length from min-volume box edge aligned with PCA (m).", GH_ParamAccess.item);
            pManager.AddNumberParameter("WidthBox", "Wb", "Width from min-volume box (m).", GH_ParamAccess.item);
            pManager.AddNumberParameter("HeightBox", "Hb", "Height from min-volume box (m).", GH_ParamAccess.item);
            pManager.AddTextParameter("Notes", "N", "Diagnostics (failed cuts, etc.).", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Brep brep = null;
            if (!DA.GetData(0, ref brep)) return;

            double tol = RhinoDoc.ActiveDoc != null
                ? RhinoDoc.ActiveDoc.ModelAbsoluteTolerance
                : 1e-4;
            tol = Math.Max(tol, 1e-9);
            if (brep != null && brep.IsValid)
            {
                double d = brep.GetBoundingBox(true).Diagonal.Length;
                if (d > 0)
                    tol = Math.Max(tol, d * 1e-6);
            }

            if (!BrepSectionFromBrep.TryRead(brep, tol, out BrepSectionReadResult res, out string err))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, err ?? "Read section failed.");
                return;
            }

            DA.SetData(0, res.PcaAxis);
            DA.SetData(1, res.LengthAlongAxis);

            if (res.PlaneCutsOk)
            {
                DA.SetData(2, res.WidthFromPlaneCuts);
                DA.SetData(3, res.HeightFromPlaneCuts);
            }
            else
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Plane-cut section could not be measured.");
                DA.SetData(2, null);
                DA.SetData(3, null);
            }

            if (res.MinVolumeBoxOk)
            {
                DA.SetData(4, res.LengthFromMinVolumeBox);
                DA.SetData(5, res.WidthFromMinVolumeBox);
                DA.SetData(6, res.HeightFromMinVolumeBox);
            }
            else
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Min-volume box dimensions unavailable.");
                DA.SetData(4, null);
                DA.SetData(5, null);
                DA.SetData(6, null);
            }

            DA.SetData(7, res.Notes ?? string.Empty);
        }
    }
}
