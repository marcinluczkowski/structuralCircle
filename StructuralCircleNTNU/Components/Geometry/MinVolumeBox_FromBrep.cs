using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Geometry
{
    /// <summary>
    /// Oriented boxes from Brep samples: heuristic minimum-volume OBB and PCA-first-axis box (Element from Brep frame).
    /// </summary>
    public class MinVolumeBox_FromBrep : GH_Component
    {
        public MinVolumeBox_FromBrep()
            : base("Min Volume Box from Brep", "MinVolBox",
                   "Minimum-volume oriented box (heuristic search over orientations) and PCA-first-axis bounding box.",
                   "StructuralCircleNTNU", "Geometry")
        { }

        public override Guid ComponentGuid => new Guid("7c2a9b10-5e4d-4f8a-9c1e-2b6d8a0e4f33");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBrepParameter("Brep", "B", "Brep to bound.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddBoxParameter("Box", "B1", "Heuristic minimum-volume oriented bounding box from samples.", GH_ParamAccess.item);
            pManager.AddBoxParameter("Box2", "B2", "Bounding box aligned with PCA axis + BuildOrthonormalFrame (same as Element from Brep).", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Brep brep = null;
            if (!DA.GetData(0, ref brep)) return;

            if (!BrepElementBuilder.TryGetSamplePoints(brep, out List<Point3d> samples, out string sampleMsg))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, sampleMsg ?? "Sampling failed.");
                return;
            }

            if (!BrepElementBuilder.TryMinimumVolumeOrientedBoxFromPoints(samples, out Box minBox, out string minMsg))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, minMsg ?? "Minimum-volume box failed.");
            }
            else
            {
                DA.SetData(0, minBox);
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                    "Minimum-volume box is heuristic (sample-based); not guaranteed globally optimal.");
            }

            if (!BrepElementBuilder.TryPcaFirstAxisBoxFromPoints(samples, out Box pcaBox, out string pcaMsg))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, pcaMsg ?? "PCA-aligned box failed.");
            }
            else
            {
                DA.SetData(1, pcaBox);
            }
        }
    }
}
