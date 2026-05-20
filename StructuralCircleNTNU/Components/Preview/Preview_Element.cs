using System;
using System.Drawing;
using Grasshopper.Kernel;
using Rhino.Geometry;
using StructuralCircleNTNU;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Preview
{
    public class Preview_Element : GH_Component
    {
        private GeometryBase _displayGeom;
        private Point3d      _labelPt;
        private string       _label;

        public Preview_Element()
            : base("Preview Element", "PrevElem",
                   "Visualise a single Beam or Plate in the Rhino viewport.",
                   "StructuralCircleNTNU", "Preview") { }

        public override Guid ComponentGuid => new Guid("78421563-f2a4-42c8-9ed1-7b9c4e5d6a01");
        protected override Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Element",        "Elem",   "Element to preview.",          GH_ParamAccess.item);
            pManager.AddPointParameter  ("Origin",         "Pt",     "Insertion point in viewport.", GH_ParamAccess.item);
            pManager.AddBooleanParameter("ShowLabel",      "Label",  "Show element name label.",     GH_ParamAccess.item);

            pManager[1].Optional = true;
            pManager[2].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGeometryParameter("Geometry", "Geo", "Display geometry.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            _displayGeom = null;
            _label = null;

            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var origin = Point3d.Origin;
            DA.GetData(1, ref origin);

            bool showLabel = true;
            DA.GetData(2, ref showLabel);

            var element = GrasshopperUnpack.AsElement(raw);
            if (element == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not an Element.");
                return;
            }

            _displayGeom = GeometryBuilder.BuildElementGeometry(element, origin);
            _labelPt     = GeometryBuilder.LabelPoint(element, origin);
            if (showLabel) _label = element.Name ?? element.Id.ToString();

            if (_displayGeom != null)
                DA.SetData(0, _displayGeom);
        }

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            if (_displayGeom is Brep brep)
                args.Display.DrawBrepWires(brep, Color.DarkSlateGray, 1);
            else if (_displayGeom is Curve crv)
                args.Display.DrawCurve(crv, Color.DarkSlateGray, 2);

            if (_label != null)
                args.Display.Draw2dText(_label, Color.Black, _labelPt, false, 12);
        }

        public override void DrawViewportMeshes(IGH_PreviewArgs args)
        {
            if (_displayGeom is Brep brep)
            {
                var mesh = Mesh.CreateFromBrep(brep, MeshingParameters.FastRenderMesh);
                if (mesh != null)
                    foreach (var m in mesh)
                        args.Display.DrawMeshShaded(m, new Rhino.Display.DisplayMaterial(Color.BurlyWood, 0.3));
            }
        }

        public override bool IsPreviewCapable => true;
    }
}
