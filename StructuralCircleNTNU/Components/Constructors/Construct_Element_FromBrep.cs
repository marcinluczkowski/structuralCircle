using System;
using Grasshopper.Kernel;
using Rhino.Geometry;
using StructuralCircleNTNU;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Constructors
{
    /// <summary>
    /// Builds a <see cref="Beam"/> or <see cref="Plate"/> from a solid Brep: PCA axis, cross-section from perpendicular
    /// plane cuts (25/50/75% along axis) with OBB fallback, automatic beam vs plate classification. Dimensions to 4 decimals.
    /// </summary>
    public class Construct_Element_FromBrep : GH_Component
    {
        public Construct_Element_FromBrep()
            : base("Element from Brep", "ElemBrep",
                   "Create a Beam or Plate from a Brep using PCA axis, plane-cut cross-section (OBB fallback), four-decimal dimensions.",
                   "StructuralCircleNTNU", "Construct")
        { }

        public override Guid ComponentGuid => new Guid("7e4f2a10-9c3d-4b71-a1e2-8f5d6c4b3a01");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBrepParameter  ("Brep",     "B",   "Closed or open Brep; surface-sampled internally for analysis.", GH_ParamAccess.item);
            pManager.AddGenericParameter("Material", "Mat", "Material for the element.",                            GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Element", "Elem", "Beam or Plate element.",           GH_ParamAccess.item);
            pManager.AddTextParameter   ("Type",    "T",    "Beam or Plate.",                   GH_ParamAccess.item);
            pManager.AddLineParameter   ("Axis",    "Ax",   "PCA structural axis.",             GH_ParamAccess.item);
            pManager.AddNumberParameter ("Length",  "L",    "Length along PCA axis (m, 4 d.p.).", GH_ParamAccess.item);
            pManager.AddNumberParameter ("Dim1",    "D1",   "Section width from plane cuts (m, 4 d.p.); OBB if cuts failed.", GH_ParamAccess.item);
            pManager.AddNumberParameter ("Dim2",    "D2",   "Section height from plane cuts (m, 4 d.p.); OBB if cuts failed.", GH_ParamAccess.item);
            pManager.AddTextParameter   ("Report",  "Info", "Summary of analysis.",             GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Brep brep = null;
            object rawMat = null;
            if (!DA.GetData(0, ref brep)) return;
            if (!DA.GetData(1, ref rawMat)) return;

            var material = GrasshopperUnpack.AsMaterial(rawMat);
            if (material == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                    "Material must be a StructuralCircleNTNU Material (e.g. from Construct Material). " +
                    "Generic wires wrap data; use the plugin Material output, not Rhino’s render Material.");
                return;
            }

            int id = material.Id;
            string name = $"FromBrep_{material.Name}".Replace(' ', '_');

            var res = BrepElementBuilder.TryBuild(brep, material, id, name, "");

            if (!res.Success || res.Element == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, res.Message ?? "Build failed.");
                return;
            }

            DA.SetData(0, res.Element);
            DA.SetData(1, res.ClassifiedAsPlate ? "Plate" : "Beam");
            DA.SetData(2, res.Axis);
            DA.SetData(3, res.LengthAlongAxis);
            DA.SetData(4, res.Extent1);
            DA.SetData(5, res.Extent2);
            DA.SetData(6, res.Message);
        }
    }
}
