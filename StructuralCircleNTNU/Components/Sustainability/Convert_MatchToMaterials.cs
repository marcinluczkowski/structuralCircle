using System;
using System.Collections.Generic;
using System.Linq;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;
using StructuralCircleNTNU.Classes.Sustainability;

namespace StructuralCircleNTNU.Components.Sustainability
{
    /// <summary>
    /// Bridges a MatchResult to the sustainability LCA pipeline.
    ///
    /// Every demand element in the project becomes one MaterialItem:
    ///   • Matched demand (covered by a reclaimed supply)  → ReuseShare = 1.0
    ///   • Unmatched demand (must be produced as new)      → ReuseShare = 0.0
    ///
    /// Material type is fixed to C24 sawn timber by default (density 420 kg/m³,
    /// A1–A3 EF 0.28 kgCO2e/kg) — override with the inputs below. The same emission
    /// factor is used for both reused and new; the LCA allocation method selected in
    /// Method_SustainabilityAnalysis controls how the reused burden is split.
    ///
    /// Outputs:
    ///   • ItemsPerElement  — one MaterialItem per demand element (detailed, use in DesignConcept)
    ///   • ReusePool        — one aggregated MaterialItem for all reused elements (ReuseShare = 1.0)
    ///   • NewPool          — one aggregated MaterialItem for all new elements    (ReuseShare = 0.0)
    ///   • Stats            — summary text (counts, volumes, masses)
    /// </summary>
    public class Convert_MatchToMaterials : GH_Component
    {
        // ── Default C24 sawn-timber constants ────────────────────────────────────
        // EN 338 mean density for C24: 420 kg/m³
        // A1–A3 EF (sawn structural softwood): 0.28 kgCO2e/kg
        //   (representative of EN 15804 cradle-to-gate EPDs for Nordic spruce/pine)
        const double DefaultDensityKgM3 = 420.0;
        const double DefaultEfA1A3      = 0.28;    // kgCO2e / kg
        const double DefaultTransportNew    = 200.0; // km for virgin timber
        const double DefaultTransportReused = 50.0;  // km for reclaimed element
        const double DefaultTransportEF = 0.1;      // kgCO2e / (tonne · km)
        const double DefaultWasteFactor = 0.05;

        public Convert_MatchToMaterials()
            : base("Match to Materials", "Match2Mat",
                   "Convert a MatchResult into MaterialItems for LCA. " +
                   "Matched demand elements get ReuseShare=1, unmatched get ReuseShare=0. " +
                   "Assumes C24 timber throughout (override with inputs).",
                   "StructuralCircleNTNU", "Sustainability")
        { }

        public override Guid ComponentGuid => new Guid("A2B4C6D8-1234-5678-9ABC-DEF012345678");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddGenericParameter ("MatchResult",     "R",       "MatchResult from a matching algorithm.",                                                       GH_ParamAccess.item);
            p.AddTextParameter    ("MaterialType",    "Type",    "Material family label, e.g. 'C24 timber', 'glulam', 'steel'.",                                 GH_ParamAccess.item, "C24 timber");
            p.AddNumberParameter  ("Density",         "rho",     "Density in kg/m³. C24 = 420 kg/m³.",                                                           GH_ParamAccess.item, DefaultDensityKgM3);
            p.AddNumberParameter  ("EF_A1A3",         "EF",      "Production emission factor in kgCO2e/kg (A1–A3).",                                             GH_ParamAccess.item, DefaultEfA1A3);
            p.AddNumberParameter  ("TransportKm_New", "Tkm_N",   "Transport distance (km) for new / virgin elements.",                                           GH_ParamAccess.item, DefaultTransportNew);
            p.AddNumberParameter  ("TransportKm_Reu", "Tkm_R",   "Transport distance (km) for reclaimed / reused elements.",                                     GH_ParamAccess.item, DefaultTransportReused);
            p.AddNumberParameter  ("TransportEF",     "TEF",     "Transport emission factor in kgCO2e/(tonne·km). Road = 0.1.",                                  GH_ParamAccess.item, DefaultTransportEF);
            p.AddNumberParameter  ("WasteFactor",     "W",       "On-site construction waste fraction (0–1).",                                                    GH_ParamAccess.item, DefaultWasteFactor);

            for (int i = 1; i < 8; i++) p[i].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddGenericParameter("ItemsPerElement", "Items", "One MaterialItem per demand element (matched first, unmatched after). Feed list into DesignConcept.", GH_ParamAccess.list);
            p.AddGenericParameter("ReusePool",       "Reuse", "One aggregated MaterialItem for all reused elements (ReuseShare=1). Handy for a quick two-line LCA.",   GH_ParamAccess.item);
            p.AddGenericParameter("NewPool",         "New",   "One aggregated MaterialItem for all new elements (ReuseShare=0).",                                       GH_ParamAccess.item);
            p.AddTextParameter   ("Stats",           "S",     "Summary: element counts, total volumes, masses and reuse ratio.",                                        GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var result = GrasshopperUnpack.AsMatchResult(raw);
            if (result == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a MatchResult.");
                return;
            }

            string matType      = "C24 timber";
            double density      = DefaultDensityKgM3;
            double ef           = DefaultEfA1A3;
            double tKmNew       = DefaultTransportNew;
            double tKmReu       = DefaultTransportReused;
            double tEF          = DefaultTransportEF;
            double wasteFactor  = DefaultWasteFactor;

            DA.GetData(1, ref matType);
            DA.GetData(2, ref density);
            DA.GetData(3, ref ef);
            DA.GetData(4, ref tKmNew);
            DA.GetData(5, ref tKmReu);
            DA.GetData(6, ref tEF);
            DA.GetData(7, ref wasteFactor);

            density     = Math.Max(density,     1e-3);
            ef          = Math.Max(ef,           0.0);
            tEF         = Math.Max(tEF,          0.0);
            wasteFactor = Math.Max(0, Math.Min(wasteFactor, 1.0));

            var pairs           = result.Pairs           ?? new List<MatchPair>();
            var unmatchedDemand = result.UnmatchedDemand ?? new List<Element>();

            var items        = new List<MaterialItem>();
            double volReused = 0, volNew = 0;
            double massReused = 0, massNew = 0;

            // ── Matched demand elements (reused from supply) ─────────────────────
            foreach (var pair in pairs)
            {
                if (pair.Demand == null) continue;

                double vol  = MeasureVolumeM3(pair.Demand);
                double mass = vol * density;

                var item = MakeItem(
                    label      : pair.Demand.Name ?? ("E" + pair.Demand.Id),
                    matType    : matType,
                    vol        : vol,
                    mass       : mass,
                    density    : density,
                    ef         : ef,
                    reuseShare : 1.0,
                    transportKm: tKmReu,
                    tEF        : tEF,
                    wasteFactor: wasteFactor,
                    note       : $"→ supply {pair.Supply?.Name ?? "?"}");

                items.Add(item);
                volReused  += vol;
                massReused += mass;
            }

            // ── Unmatched demand elements (new material needed) ──────────────────
            foreach (var elem in unmatchedDemand)
            {
                if (elem == null) continue;

                double vol  = MeasureVolumeM3(elem);
                double mass = vol * density;

                var item = MakeItem(
                    label      : elem.Name ?? ("E" + elem.Id),
                    matType    : matType,
                    vol        : vol,
                    mass       : mass,
                    density    : density,
                    ef         : ef,
                    reuseShare : 0.0,
                    transportKm: tKmNew,
                    tEF        : tEF,
                    wasteFactor: wasteFactor,
                    note       : "new");

                items.Add(item);
                volNew  += vol;
                massNew += mass;
            }

            // ── Aggregate pools ─────────────────────────────────────────────────
            MaterialItem reusePool = null;
            if (massReused > 0)
            {
                reusePool = MakeItem(
                    label      : $"Reused {matType} ({pairs.Count} elements)",
                    matType    : matType,
                    vol        : volReused,
                    mass       : massReused,
                    density    : density,
                    ef         : ef,
                    reuseShare : 1.0,
                    transportKm: tKmReu,
                    tEF        : tEF,
                    wasteFactor: wasteFactor,
                    note       : null);
            }

            MaterialItem newPool = null;
            if (massNew > 0)
            {
                newPool = MakeItem(
                    label      : $"New {matType} ({unmatchedDemand.Count} elements)",
                    matType    : matType,
                    vol        : volNew,
                    mass       : massNew,
                    density    : density,
                    ef         : ef,
                    reuseShare : 0.0,
                    transportKm: tKmNew,
                    tEF        : tEF,
                    wasteFactor: wasteFactor,
                    note       : null);
            }

            // ── Stats ─────────────────────────────────────────────────────────────
            int totalElems  = pairs.Count + unmatchedDemand.Count;
            double totalVol = volReused + volNew;
            double totalMass = massReused + massNew;
            double reuseRatio = totalMass > 1e-9 ? massReused / totalMass : 0.0;

            var stats = new System.Text.StringBuilder();
            stats.AppendLine($"Match→Materials ({matType})");
            stats.AppendLine($"  Demand elements  : {totalElems}");
            stats.AppendLine($"  Reused (matched) : {pairs.Count}  — {volReused:F4} m³  /  {massReused:F1} kg");
            stats.AppendLine($"  New (unmatched)  : {unmatchedDemand.Count}  — {volNew:F4} m³  /  {massNew:F1} kg");
            stats.AppendLine($"  Total volume     : {totalVol:F4} m³");
            stats.AppendLine($"  Total mass       : {totalMass:F1} kg");
            stats.AppendLine($"  Reuse ratio (kg) : {reuseRatio * 100:F1} %");
            stats.AppendLine($"  Density          : {density:F0} kg/m³");
            stats.AppendLine($"  EF A1–A3         : {ef:F3} kgCO2e/kg");
            stats.AppendLine($"  Transport new    : {tKmNew:F0} km @ {tEF:F3} kgCO2e/(t·km)");
            stats.AppendLine($"  Transport reused : {tKmReu:F0} km @ {tEF:F3} kgCO2e/(t·km)");

            if (pairs.Count > 0 && unmatchedDemand.Count == 0)
                stats.AppendLine("  ✓ All demand fully covered by supply (100 % reuse scenario).");
            else if (pairs.Count == 0)
                stats.AppendLine("  ⚠ No elements matched — 100 % new material scenario.");

            DA.SetDataList(0, items);
            DA.SetData    (1, reusePool);
            DA.SetData    (2, newPool);
            DA.SetData    (3, stats.ToString());
        }

        // ── Volume calculation ────────────────────────────────────────────────────
        /// <summary>
        /// Returns the volume of the demand element in m³.
        /// Priority: section dimensions > GeometryBrep BBox.
        /// Beam   : length × section area (= cross-sectional area from BeamSection).
        /// Plate  : length × width × thickness (from PlateSection).
        /// Fallback: axis length × unit cross-section (1 cm²) with a warning flag.
        /// </summary>
        static double MeasureVolumeM3(Element elem)
        {
            if (elem is Beam beam)
            {
                double L = beam.Length;
                if (L > 1e-9 && beam.Section is BeamSection bs && bs.Area > 1e-9)
                    return L * bs.Area;

                if (beam.GeometryBrep != null && beam.GeometryBrep.IsValid)
                    return BBoxVolume(beam.GeometryBrep);

                return L * 0.01 * 0.01; // 10×10 cm fallback
            }

            if (elem is Plate plate)
            {
                double L = plate.Length;
                if (L > 1e-9 && plate.Section is PlateSection ps && ps.Thickness > 1e-9)
                {
                    double w = ps.Width > 1e-9 ? ps.Width : L; // square fallback
                    return L * w * ps.Thickness;
                }

                if (plate.GeometryBrep != null && plate.GeometryBrep.IsValid)
                    return BBoxVolume(plate.GeometryBrep);

                return L * L * 0.02; // thin slab fallback
            }

            if (elem.GeometryBrep != null && elem.GeometryBrep.IsValid)
                return BBoxVolume(elem.GeometryBrep);

            return elem.AxisLine.IsValid ? elem.AxisLine.Length * 0.01 * 0.01 : 0;
        }

        static double BBoxVolume(Rhino.Geometry.Brep brep)
        {
            var bb = brep.GetBoundingBox(true);
            if (!bb.IsValid) return 0;
            return (bb.Max.X - bb.Min.X) * (bb.Max.Y - bb.Min.Y) * (bb.Max.Z - bb.Min.Z);
        }

        // ── Factory helper ──────────────────────────────────────────────────────
        static MaterialItem MakeItem(
            string label, string matType,
            double vol, double mass, double density,
            double ef, double reuseShare,
            double transportKm, double tEF, double wasteFactor,
            string note)
        {
            string fullName = string.IsNullOrEmpty(note)
                ? label
                : $"{label} [{note}]";

            return new MaterialItem
            {
                Name                     = fullName,
                MaterialType             = matType,
                QuantityM3               = vol,
                QuantityKg               = mass,
                DensityKgM3              = density,
                Unit                     = "kg",
                EmissionFactorA1A3       = ef,
                ReuseShare               = reuseShare < 0 ? 0 : (reuseShare > 1 ? 1 : reuseShare),
                TransportDistanceKm      = transportKm,
                TransportFactorKgCO2ePerTkm = tEF,
                WasteFactor              = wasteFactor
            };
        }
    }
}
