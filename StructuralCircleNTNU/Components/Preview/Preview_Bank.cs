using System;
using System.Collections.Generic;
using System.Drawing;
using Grasshopper.Kernel;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Preview
{
    public class Preview_Bank : GH_Component
    {
        private readonly List<GeometryBase> _geometries = new List<GeometryBase>();
        private readonly List<string>       _labels     = new List<string>();
        private readonly List<Point3d>      _labelPts   = new List<Point3d>();
        private readonly List<bool>         _isBeam     = new List<bool>();

        public Preview_Bank()
            : base("Preview Bank", "PrevBank",
                   "Lay out all elements from a SupplyBank or DemandBank along the X-axis.",
                   "StructuralCircleNTNU", "Preview") { }

        public override Guid ComponentGuid => new Guid("8a531674-03b5-43d9-af82-8cad5f6e7b12");
        protected override Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Bank",      "Bank",    "SupplyBank or DemandBank.",     GH_ParamAccess.item);
            pManager.AddPointParameter  ("Origin",    "Pt",      "Start insertion point.",         GH_ParamAccess.item);
            pManager.AddNumberParameter ("SpacingX",  "dX",      "Spacing between elements (m).",  GH_ParamAccess.item);
            pManager.AddBooleanParameter("ShowLabels","Labels",   "Show element name labels.",      GH_ParamAccess.item);

            pManager[1].Optional = true;
            pManager[2].Optional = true;
            pManager[3].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGeometryParameter("Geometries", "Geo", "All display geometries.", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            _geometries.Clear();
            _labels.Clear();
            _labelPts.Clear();
            _isBeam.Clear();

            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var origin   = Point3d.Origin;
            var spacingX = 0.5;
            var showLabels = true;
            DA.GetData(1, ref origin);
            DA.GetData(2, ref spacingX);
            DA.GetData(3, ref showLabels);

            List<Element> elements = null;
            if (raw is SupplyBank supply) elements = supply.Elements;
            else if (raw is DemandBank demand) elements = demand.Elements;
            else { AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input must be a SupplyBank or DemandBank."); return; }

            double cursorX = origin.X;

            foreach (var elem in elements)
            {
                var pt   = new Point3d(cursorX, origin.Y, origin.Z);
                var geom = GeometryBuilder.BuildElementGeometry(elem, pt);
                _geometries.Add(geom);
                _isBeam.Add(elem is Beam);

                if (showLabels)
                {
                    _labels.Add(elem.Name ?? elem.Id.ToString());
                    _labelPts.Add(GeometryBuilder.LabelPoint(elem, pt));
                }
                else
                {
                    _labels.Add(null);
                    _labelPts.Add(Point3d.Unset);
                }

                double elemLen = elem is Beam b ? b.Length
                               : elem is Plate p ? p.Length
                               : 0;
                cursorX += elemLen + spacingX;
            }

            var output = new List<GeometryBase>();
            foreach (var g in _geometries) if (g != null) output.Add(g);
            DA.SetDataList(0, output);
        }

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            for (int i = 0; i < _geometries.Count; i++)
            {
                var g = _geometries[i];
                if (g == null) continue;

                var col = _isBeam[i] ? Color.SaddleBrown : Color.Tan;

                if (g is Brep brep)
                    args.Display.DrawBrepWires(brep, col, 1);
                else if (g is Curve crv)
                    args.Display.DrawCurve(crv, col, 2);

                if (_labels[i] != null)
                    args.Display.Draw2dText(_labels[i], Color.Black, _labelPts[i], false, 11);
            }
        }

        public override void DrawViewportMeshes(IGH_PreviewArgs args)
        {
            for (int i = 0; i < _geometries.Count; i++)
            {
                if (!(_geometries[i] is Brep brep)) continue;
                var col = _isBeam[i] ? Color.SaddleBrown : Color.Tan;
                var mat = new Rhino.Display.DisplayMaterial(col, 0.25);
                var meshes = Mesh.CreateFromBrep(brep, MeshingParameters.FastRenderMesh);
                if (meshes == null) continue;
                foreach (var m in meshes)
                    args.Display.DrawMeshShaded(m, mat);
            }
        }

        public override bool IsPreviewCapable => true;
    }
}
