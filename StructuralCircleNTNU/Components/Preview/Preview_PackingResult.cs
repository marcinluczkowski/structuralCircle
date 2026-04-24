using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Grasshopper.Kernel;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Preview
{
    /// <summary>
    /// Visualises how each supply element is cut to produce its matched demand pieces.
    /// Only used supplies are drawn (those carrying at least one packed demand). Supplies
    /// are stacked along +World Y in canonical preview frame (member along +Z, shorter
    /// cross-section on ±X, longer cross-section on ±Y). Each demand is placed at its
    /// exact packing offset inside the supply, with a label at the centre of its BBox.
    /// Supplies are drawn as wireframe outlines; demands as semi-transparent shaded
    /// volumes so the cut positions are visible through the stock.
    /// </summary>
    public class Preview_PackingResult : GH_Component
    {
        // Parallel lists rebuilt on every solve; used by the custom viewport drawing below.
        private readonly List<GeometryBase> _supplyGeoms  = new List<GeometryBase>();
        private readonly List<GeometryBase> _demandGeoms  = new List<GeometryBase>();
        private readonly List<Point3d>      _supplyLabelPts = new List<Point3d>();
        private readonly List<string>       _supplyLabels   = new List<string>();
        private readonly List<Point3d>      _demandLabelPts = new List<Point3d>();
        private readonly List<string>       _demandLabels   = new List<string>();

        public Preview_PackingResult()
            : base("Preview Packing Result", "PrevPack",
                   "Visualise cut layout of a packed MatchResult: every used supply drawn once " +
                   "with its demand pieces placed at the exact packing offset. " +
                   "Only supplies carrying matched demand are previewed. Labels sit at the centre " +
                   "of each demand BBox so the cut plan can be read off directly.",
                   "StructuralCircleNTNU", "Preview")
        { }

        public override Guid ComponentGuid => new Guid("7C3A2F10-9B4E-4D52-A0B8-1F6E8D3C5A90");
        protected override Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("MatchResult", "R",
                "MatchResult from a packing matching algorithm (modes 3 or 4).",
                GH_ParamAccess.item);
            pManager.AddPointParameter("Origin", "Pt",
                "Layout origin: base of the first (lowest-Y) supply element.",
                GH_ParamAccess.item, Point3d.Origin);
            pManager.AddNumberParameter("Gap", "Gap_Y",
                "Extra +Y spacing between consecutive supplies.",
                GH_ParamAccess.item, 0.2);
            pManager.AddBooleanParameter("ShowLabels", "L",
                "Show supply and demand name labels.",
                GH_ParamAccess.item, true);

            pManager[1].Optional = true;
            pManager[2].Optional = true;
            pManager[3].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGeometryParameter("SupplyGeometries", "SGeo",
                "Used supply geometry (one per unique supply element).",
                GH_ParamAccess.list);
            pManager.AddGeometryParameter("DemandGeometries", "DGeo",
                "Demand geometry placed at each cut position inside its supply.",
                GH_ParamAccess.list);
            pManager.AddPointParameter("LabelPoints", "Pts",
                "Label anchor points (supply labels first, then demand labels).",
                GH_ParamAccess.list);
            pManager.AddTextParameter("Labels", "Txt",
                "Label text, parallel to LabelPoints.",
                GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            _supplyGeoms.Clear();
            _demandGeoms.Clear();
            _supplyLabelPts.Clear();
            _supplyLabels.Clear();
            _demandLabelPts.Clear();
            _demandLabels.Clear();

            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var result = GrasshopperUnpack.AsMatchResult(raw);
            if (result == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a MatchResult.");
                return;
            }

            Point3d origin = Point3d.Origin;
            double gap = 0.2;
            bool showLabels = true;
            DA.GetData(1, ref origin);
            DA.GetData(2, ref gap);
            DA.GetData(3, ref showLabels);

            var pairs = result.Pairs ?? new List<MatchPair>();
            if (pairs.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "MatchResult has no pairs to preview.");
                return;
            }

            var order  = new List<Element>();
            var groups = new Dictionary<Element, List<MatchPair>>();
            foreach (var p in pairs)
            {
                if (p.Supply == null) continue;
                if (!groups.ContainsKey(p.Supply))
                {
                    order.Add(p.Supply);
                    groups[p.Supply] = new List<MatchPair>();
                }
                groups[p.Supply].Add(p);
            }

            if (order.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No valid supply elements in MatchResult.");
                return;
            }

            bool anyPlacement = pairs.Any(p => p.HasPlacement);
            if (!anyPlacement)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                    "No packing placements on pairs — showing demand centred on each supply's mid-length. " +
                    "Use matching modes 3 (GreedyPacking) or 4 (MilpPacking) to get exact cut offsets.");

            double cursorY = origin.Y;
            foreach (var supply in order)
            {
                var supplyPairs = groups[supply];

                var supplyGeom = PreviewBankLayout.BuildGeometry(supply);
                if (supplyGeom == null) continue;

                var supplyBB = supplyGeom.GetBoundingBox(true);
                if (!supplyBB.IsValid) continue;

                var supplyBase = new Point3d(origin.X, cursorY, origin.Z);

                var supplyPlaced = supplyGeom.Duplicate();
                supplyPlaced.Transform(Transform.Translation(new Vector3d(supplyBase)));
                _supplyGeoms.Add(supplyPlaced);

                if (showLabels)
                {
                    double supplyZMid = supplyBB.Min.Z + 0.5 * (supplyBB.Max.Z - supplyBB.Min.Z);
                    _supplyLabelPts.Add(new Point3d(supplyBase.X, supplyBase.Y, supplyBase.Z + supplyZMid));
                    _supplyLabels.Add(supply.Name ?? ("Supply " + supply.Id));
                }

                // Axis mapping between packing frame (DX=length, DY=width, DZ=height) and the
                // preview frame (X=shortCross, Y=longCross, Z=length). Length always maps 1:1;
                // width/height swap depending on which is the longer cross-section dim.
                var supplyPack = PackingEngine.GetBbox(supply);
                bool packYisPreviewY = supplyPack.DY >= supplyPack.DZ;

                double supplyShort = Math.Min(supplyPack.DY, supplyPack.DZ);
                double supplyLong  = Math.Max(supplyPack.DY, supplyPack.DZ);

                double midZ = 0.5 * (supplyBB.Max.Z - supplyBB.Min.Z);

                foreach (var pair in supplyPairs)
                {
                    var demand = pair.Demand;
                    if (demand == null) continue;

                    var demandGeom = PreviewBankLayout.BuildGeometry(demand);
                    if (demandGeom == null) continue;

                    var demandBB = demandGeom.GetBoundingBox(true);
                    if (!demandBB.IsValid) continue;

                    var demandPack = PackingEngine.GetBbox(demand);
                    double demandShort = Math.Min(demandPack.DY, demandPack.DZ);
                    double demandLong  = Math.Max(demandPack.DY, demandPack.DZ);

                    double offX = 0;
                    double offY = 0;
                    double offZ = midZ - 0.5 * (demandBB.Max.Z - demandBB.Min.Z); // fallback: centered

                    if (pair.HasPlacement)
                    {
                        double packX = pair.Placement.From.X;          // along length
                        double packY = pair.Placement.From.Y;          // shelf-width or BBox Y
                        double packZ = pair.Placement.From.Z;          // BBox Z (height)

                        offZ = packX;                                   // length axis 1:1

                        if (packYisPreviewY)
                        {
                            offY = packY - 0.5 * supplyLong  + 0.5 * demandLong;
                            offX = packZ - 0.5 * supplyShort + 0.5 * demandShort;
                        }
                        else
                        {
                            offY = packZ - 0.5 * supplyLong  + 0.5 * demandLong;
                            offX = packY - 0.5 * supplyShort + 0.5 * demandShort;
                        }
                    }

                    var t = new Vector3d(supplyBase.X + offX, supplyBase.Y + offY, supplyBase.Z + offZ);
                    var demandPlaced = demandGeom.Duplicate();
                    demandPlaced.Transform(Transform.Translation(t));
                    _demandGeoms.Add(demandPlaced);

                    if (showLabels)
                    {
                        var placedBB = demandPlaced.GetBoundingBox(true);
                        _demandLabelPts.Add(new Point3d(
                            0.5 * (placedBB.Min.X + placedBB.Max.X),
                            0.5 * (placedBB.Min.Y + placedBB.Max.Y),
                            0.5 * (placedBB.Min.Z + placedBB.Max.Z)));
                        _demandLabels.Add(demand.Name ?? ("Demand " + demand.Id));
                    }
                }

                double stride = PreviewBankLayout.GetStrideAlongY(supply);
                cursorY += stride + gap;
            }

            DA.SetDataList(0, _supplyGeoms.Select(g => g));
            DA.SetDataList(1, _demandGeoms.Select(g => g));

            var allPts    = _supplyLabelPts.Concat(_demandLabelPts).ToList();
            var allLabels = _supplyLabels.Concat(_demandLabels).ToList();
            DA.SetDataList(2, allPts);
            DA.SetDataList(3, allLabels);
        }

        // ── Viewport drawing ─────────────────────────────────────────────────────
        static readonly Color ColSupplyWire   = Color.FromArgb( 80,  80,  80);  // dark grey
        static readonly Color ColDemandFill   = Color.FromArgb(60, 180,  80);   // green
        static readonly Color ColDemandWire   = Color.FromArgb(30, 130,  50);   // dark green
        static readonly Color ColSupplyLabel  = Color.FromArgb( 40,  40,  40);
        static readonly Color ColDemandLabel  = Color.FromArgb( 20,  90,  30);

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            foreach (var g in _supplyGeoms)
            {
                if (g is Brep b) args.Display.DrawBrepWires(b, ColSupplyWire, 2);
                else if (g is Curve c) args.Display.DrawCurve(c, ColSupplyWire, 2);
            }

            foreach (var g in _demandGeoms)
            {
                if (g is Brep b) args.Display.DrawBrepWires(b, ColDemandWire, 1);
                else if (g is Curve c) args.Display.DrawCurve(c, ColDemandWire, 1);
            }

            for (int i = 0; i < _supplyLabels.Count; i++)
                if (_supplyLabelPts[i].IsValid)
                    args.Display.Draw2dText(_supplyLabels[i], ColSupplyLabel, _supplyLabelPts[i], false, 11);

            for (int i = 0; i < _demandLabels.Count; i++)
                if (_demandLabelPts[i].IsValid)
                    args.Display.Draw2dText(_demandLabels[i], ColDemandLabel, _demandLabelPts[i], true, 10);
        }

        public override void DrawViewportMeshes(IGH_PreviewArgs args)
        {
            // Supplies are drawn wire-only so the demand pieces nested inside remain visible.
            // Demands drawn with a semi-transparent material.
            var mat = new Rhino.Display.DisplayMaterial(ColDemandFill, 0.45);
            foreach (var g in _demandGeoms)
            {
                if (!(g is Brep b)) continue;
                var meshes = Mesh.CreateFromBrep(b, MeshingParameters.FastRenderMesh);
                if (meshes == null) continue;
                foreach (var m in meshes)
                    args.Display.DrawMeshShaded(m, mat);
            }
        }

        public override bool IsPreviewCapable => true;
    }
}
