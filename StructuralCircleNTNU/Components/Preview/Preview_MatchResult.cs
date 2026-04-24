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
    /// Visualises a MatchResult: demand elements in the left column, supply in the right column,
    /// both laid out using the same canonical PreviewBankLayout (member along +Y, shorter
    /// cross-section dim on ±X, longer on +Z). Matched pairs are connected with amber lines.
    /// Color legend:
    ///   Blue   – matched demand
    ///   Red    – unmatched demand
    ///   Green  – matched supply
    ///   Grey   – unmatched supply
    ///   Amber  – match connection lines
    /// </summary>
    public class Preview_MatchResult : GH_Component
    {
        private readonly List<GeometryBase> _demandGeoms   = new List<GeometryBase>();
        private readonly List<GeometryBase> _supplyGeoms   = new List<GeometryBase>();
        private readonly List<bool>         _demandMatched = new List<bool>();
        private readonly List<bool>         _supplyMatched = new List<bool>();
        private readonly List<Line>         _matchLines    = new List<Line>();
        private readonly List<string>       _labels        = new List<string>();
        private readonly List<Point3d>      _labelPts      = new List<Point3d>();

        public Preview_MatchResult()
            : base("Preview Match Result", "PrevMatch",
                   "Visualise matching results: demand (left) and supply (right) side by side, " +
                   "connected matched pairs with lines. " +
                   "Colour: blue=matched demand, red=unmatched demand, green=matched supply, grey=unmatched supply.",
                   "StructuralCircleNTNU", "Preview") { }

        public override Guid ComponentGuid => new Guid("4A7B9C2D-1E3F-4A5B-8C9D-0E1F2A3B4C5D");
        protected override Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("MatchResult",  "R",       "MatchResult from a matching algorithm.",                             GH_ParamAccess.item);
            pManager.AddPointParameter  ("Origin",       "Pt",      "Layout start point (foot of demand column).",                        GH_ParamAccess.item);
            pManager.AddNumberParameter ("ColumnGap",    "Gap_X",   "Horizontal X distance between demand and supply columns. Default 2.", GH_ParamAccess.item, 2.0);
            pManager.AddNumberParameter ("ElementGap",   "Gap_Y",   "Extra Y spacing between consecutive elements in each column.",        GH_ParamAccess.item, 0.0);
            pManager.AddBooleanParameter("ShowLabels",   "Labels",  "Show element name labels.",                                          GH_ParamAccess.item, true);

            pManager[1].Optional = true;
            pManager[2].Optional = true;
            pManager[3].Optional = true;
            pManager[4].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGeometryParameter("DemandGeometries", "DGeo",  "Demand element geometries (matched first, then unmatched).", GH_ParamAccess.list);
            pManager.AddGeometryParameter("SupplyGeometries", "SGeo",  "Supply element geometries (matched first, then unmatched).", GH_ParamAccess.list);
            pManager.AddLineParameter    ("MatchLines",       "Lines", "Lines connecting each matched demand/supply pair.",           GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            _demandGeoms.Clear();
            _supplyGeoms.Clear();
            _demandMatched.Clear();
            _supplyMatched.Clear();
            _matchLines.Clear();
            _labels.Clear();
            _labelPts.Clear();

            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var result = GrasshopperUnpack.AsMatchResult(raw);
            if (result == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a MatchResult.");
                return;
            }

            var origin     = Point3d.Origin;
            double colGap  = 2.0;
            double elemGap = 0.0;
            bool showLabels = true;
            DA.GetData(1, ref origin);
            DA.GetData(2, ref colGap);
            DA.GetData(3, ref elemGap);
            DA.GetData(4, ref showLabels);

            var pairs           = result.Pairs ?? new List<MatchPair>();
            var unmatchedDemand = result.UnmatchedDemand ?? new List<Element>();
            var unmatchedSupply = result.UnmatchedSupply ?? new List<Element>();
            int nPairs          = pairs.Count;

            // Matched elements first so index < nPairs means "matched" in the parallel lists.
            var allDemand = pairs.Select(p => p.Demand).Concat(unmatchedDemand).ToList();
            var allSupply = pairs.Select(p => p.Supply).Concat(unmatchedSupply).ToList();

            double demandX = origin.X;
            double supplyX = origin.X + colGap;

            var demandCenters = PlaceColumn(allDemand, demandX, origin.Y, origin.Z, elemGap,
                                            showLabels, _demandGeoms, _labels, _labelPts);
            for (int i = 0; i < allDemand.Count; i++)
                _demandMatched.Add(i < nPairs);

            var supplyCenters = PlaceColumn(allSupply, supplyX, origin.Y, origin.Z, elemGap,
                                            showLabels, _supplyGeoms, _labels, _labelPts);
            for (int i = 0; i < allSupply.Count; i++)
                _supplyMatched.Add(i < nPairs);

            for (int i = 0; i < nPairs; i++)
            {
                if (i >= demandCenters.Count || i >= supplyCenters.Count) break;
                _matchLines.Add(new Line(demandCenters[i], supplyCenters[i]));
            }

            DA.SetDataList(0, _demandGeoms.Select(g => g));
            DA.SetDataList(1, _supplyGeoms.Select(g => g));
            DA.SetDataList(2, _matchLines.Select(l => l));
        }

        /// <summary>
        /// Builds one column of elements.
        /// Elements are stacked along +Y from (colX, startY, startZ).
        /// Returns mid-point (centroid Y + half depthZ) for each element — used for line endpoints.
        /// </summary>
        static List<Point3d> PlaceColumn(
            List<Element> elements,
            double colX, double startY, double startZ, double extraGap,
            bool showLabels,
            List<GeometryBase> geomOut,
            List<string>       labelsOut,
            List<Point3d>      labelPtsOut)
        {
            var centers   = new List<Point3d>();
            double cursor = 0;

            foreach (var elem in elements)
            {
                var placement = new Point3d(colX, startY + cursor, startZ);

                var geom = PreviewBankLayout.BuildGeometry(elem);
                if (geom != null)
                {
                    var dup = geom.Duplicate();
                    dup.Transform(Transform.Translation(new Vector3d(placement)));
                    geomOut.Add(dup);
                }
                else
                    geomOut.Add(null);

                PreviewBankLayout.TryGetLayoutExtents(elem, out double lenY, out double dimZ);
                double stride = PreviewBankLayout.GetStrideAlongY(elem);

                // Line endpoint: horizontal face of element (X = colX, Y at mid-length, Z at mid-depth)
                centers.Add(new Point3d(colX, startY + cursor + stride * 0.5, startZ + dimZ * 0.5));

                if (showLabels)
                {
                    labelsOut.Add(elem.Name ?? elem.Id.ToString());
                    labelPtsOut.Add(PreviewBankLayout.GetLabelPoint(placement, lenY, dimZ));
                }

                cursor += stride + extraGap;
            }

            return centers;
        }

        // ── Color scheme ────────────────────────────────────────────────────────
        static readonly Color ColMatchedDemand    = Color.FromArgb(60,  130, 220);  // blue
        static readonly Color ColUnmatchedDemand  = Color.FromArgb(210,  60,  60);  // red
        static readonly Color ColMatchedSupply    = Color.FromArgb(60,  180,  80);  // green
        static readonly Color ColUnmatchedSupply  = Color.FromArgb(155, 155, 155);  // grey
        static readonly Color ColMatchLine        = Color.FromArgb(240, 160,  30);  // amber

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            DrawColumn(args, _demandGeoms, _demandMatched, ColMatchedDemand, ColUnmatchedDemand, wireOnly: true);
            DrawColumn(args, _supplyGeoms, _supplyMatched, ColMatchedSupply, ColUnmatchedSupply, wireOnly: true);

            foreach (var ln in _matchLines)
                args.Display.DrawLine(ln, ColMatchLine, 2);

            for (int i = 0; i < _labels.Count; i++)
                if (_labels[i] != null && _labelPts[i].IsValid)
                    args.Display.Draw2dText(_labels[i], Color.Black, _labelPts[i], false, 10);
        }

        public override void DrawViewportMeshes(IGH_PreviewArgs args)
        {
            DrawColumn(args, _demandGeoms, _demandMatched, ColMatchedDemand, ColUnmatchedDemand, wireOnly: false);
            DrawColumn(args, _supplyGeoms, _supplyMatched, ColMatchedSupply, ColUnmatchedSupply, wireOnly: false);
        }

        static void DrawColumn(
            IGH_PreviewArgs args,
            List<GeometryBase> geoms,
            List<bool> matched,
            Color colMatch, Color colUnmatch,
            bool wireOnly)
        {
            for (int i = 0; i < geoms.Count; i++)
            {
                var g   = geoms[i];
                if (g == null) continue;
                bool m  = i < matched.Count && matched[i];
                var col = m ? colMatch : colUnmatch;

                if (wireOnly)
                {
                    if (g is Brep b)  args.Display.DrawBrepWires(b, col, 2);
                    else if (g is Curve c) args.Display.DrawCurve(c, col, 2);
                }
                else
                {
                    if (g is Brep b)
                    {
                        var mat    = new Rhino.Display.DisplayMaterial(col, 0.3);
                        var meshes = Mesh.CreateFromBrep(b, MeshingParameters.FastRenderMesh);
                        if (meshes == null) continue;
                        foreach (var mesh in meshes)
                            args.Display.DrawMeshShaded(mesh, mat);
                    }
                }
            }
        }

        public override bool IsPreviewCapable => true;
    }
}
