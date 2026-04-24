using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes.Sustainability;

namespace StructuralCircleNTNU.Deconstructors
{
    public class Deconstruct_SustainabilityResult : GH_Component
    {
        public Deconstruct_SustainabilityResult()
            : base("Deconstruct Sustainability Result", "DeconLCA",
                   "Explode a SustainabilityResult into all its module / metric outputs.",
                   "StructuralCircleNTNU", "Deconstructors") { }

        public override Guid ComponentGuid => new Guid("E1F2A3B4-0001-4000-8000-000000000005");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddGenericParameter("Result", "R", "SustainabilityResult to deconstruct.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddTextParameter  ("Concept",               "N",    "Concept name.",                                  GH_ParamAccess.item);
            p.AddTextParameter  ("Allocation",            "Alloc","Allocation method used.",                        GH_ParamAccess.item);
            p.AddNumberParameter("A1A3",                  "A13",  "Production emissions (kgCO2e).",                 GH_ParamAccess.item);
            p.AddNumberParameter("A4",                    "A4",   "Transport emissions (kgCO2e).",                  GH_ParamAccess.item);
            p.AddNumberParameter("A5",                    "A5",   "Construction / waste (kgCO2e).",                 GH_ParamAccess.item);
            p.AddNumberParameter("C1C4",                  "C",    "End-of-life (kgCO2e).",                          GH_ParamAccess.item);
            p.AddNumberParameter("ModuleD",               "D",    "Module D — reported separately (kgCO2e).",       GH_ParamAccess.item);
            p.AddNumberParameter("TotalA1A5",             "A1-5", "Sum A1–A5 (kgCO2e).",                            GH_ParamAccess.item);
            p.AddNumberParameter("TotalA1C4",             "A1-C4","Sum A1–C4 (kgCO2e).",                            GH_ParamAccess.item);
            p.AddNumberParameter("TotalWithD",            "+D",   "Optional A1–C4 + D (kgCO2e).",                   GH_ParamAccess.item);
            p.AddNumberParameter("CO2PerM2",              "/m²",  "CO2e per functional unit area (kgCO2e/m²).",     GH_ParamAccess.item);
            p.AddNumberParameter("CO2PerYear",            "/yr",  "CO2e per reference service year (kgCO2e/yr).",  GH_ParamAccess.item);
            p.AddNumberParameter("ReuseRatio",            "Reuse","Share of materials that are reused (0–1).",      GH_ParamAccess.item);
            p.AddNumberParameter("UtilisationRatio",      "U",    "Average structural utilisation ratio (0–1).",   GH_ParamAccess.item);
            p.AddNumberParameter("ReclamationPotential",  "RP",   "Share of mass available for future reuse (0–1).",GH_ParamAccess.item);
            p.AddTextParameter  ("InterlockingBreakEven", "IL",   "How many interlocking connections beat steel.",  GH_ParamAccess.item);
            p.AddTextParameter  ("Report",                "Rpt",  "Full human-readable report.",                    GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var r = GrasshopperUnpack.AsSustainabilityResult(raw);
            if (r == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a SustainabilityResult.");
                return;
            }

            DA.SetData(0,  r.ConceptName);
            DA.SetData(1,  r.AllocationMethod);
            DA.SetData(2,  r.A1A3);
            DA.SetData(3,  r.A4);
            DA.SetData(4,  r.A5);
            DA.SetData(5,  r.C1C4);
            DA.SetData(6,  r.ModuleD);
            DA.SetData(7,  r.TotalA1A5);
            DA.SetData(8,  r.TotalA1C4);
            DA.SetData(9,  r.OptionalTotalWithD);
            DA.SetData(10, r.CO2PerM2);
            DA.SetData(11, r.CO2PerServiceYear);
            DA.SetData(12, r.ReuseRatio);
            DA.SetData(13, r.UtilisationRatio);
            DA.SetData(14, r.ReclamationPotential);
            DA.SetData(15, r.InterlockingBreakEven);
            DA.SetData(16, r.ToString());
        }
    }
}
