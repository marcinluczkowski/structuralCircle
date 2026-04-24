using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes.Sustainability;

namespace StructuralCircleNTNU.Components.Sustainability
{
    public class Construct_MaterialItem : GH_Component
    {
        public Construct_MaterialItem()
            : base("Material Item", "MatItem",
                   "Create one material line for a DesignConcept (LCA input).",
                   "StructuralCircleNTNU", "Sustainability") { }

        public override Guid ComponentGuid => new Guid("E1F2A3B4-0001-4000-8000-000000000001");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddTextParameter  ("Name",        "N",     "Material name.",                                          GH_ParamAccess.item, "");
            p.AddTextParameter  ("MaterialType","Type",  "Material family (CLT, glulam, timber, steel, …).",        GH_ParamAccess.item, "timber");
            p.AddNumberParameter("QuantityKg",  "kg",    "Mass in kilograms.",                                      GH_ParamAccess.item, 0.0);
            p.AddNumberParameter("QuantityM3",  "m3",    "Volume in cubic metres.",                                 GH_ParamAccess.item, 0.0);
            p.AddNumberParameter("Density",     "rho",   "Density (kg/m³). Used to fill missing kg or m³.",         GH_ParamAccess.item, 500.0);
            p.AddTextParameter  ("Unit",        "U",     "Unit used for A1–A3 factor: 'kg' or 'm3'.",              GH_ParamAccess.item, "kg");
            p.AddNumberParameter("EF_A1A3",     "EF",    "Emission factor for production (kgCO2e per Unit).",       GH_ParamAccess.item, 0.35);
            p.AddNumberParameter("ReuseShare",  "Reuse", "Share of this item that is reused (0–1).",               GH_ParamAccess.item, 0.0);
            p.AddNumberParameter("TransportKm", "Tkm",   "Transport distance in kilometres.",                       GH_ParamAccess.item, 100.0);
            p.AddNumberParameter("TransportEF", "TEF",   "Transport factor (kgCO2e per tonne-km).",                GH_ParamAccess.item, 0.1);
            p.AddNumberParameter("WasteFactor", "W",     "Construction waste fraction (0–1).",                      GH_ParamAccess.item, 0.05);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddGenericParameter("MaterialItem", "M", "Material item.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            string name = "", type = "timber", unit = "kg";
            double qKg = 0, qM3 = 0, rho = 500, ef = 0.35, reuse = 0;
            double tKm = 100, tEf = 0.1, wF = 0.05;
            DA.GetData(0, ref name); DA.GetData(1, ref type);
            DA.GetData(2, ref qKg); DA.GetData(3, ref qM3); DA.GetData(4, ref rho);
            DA.GetData(5, ref unit); DA.GetData(6, ref ef);
            DA.GetData(7, ref reuse);
            DA.GetData(8, ref tKm); DA.GetData(9, ref tEf); DA.GetData(10, ref wF);

            if (qKg <= 0 && qM3 > 0 && rho > 0) qKg = qM3 * rho;
            if (qM3 <= 0 && qKg > 0 && rho > 0) qM3 = qKg / rho;

            var m = new MaterialItem
            {
                Name = name,
                MaterialType = type,
                QuantityKg = qKg,
                QuantityM3 = qM3,
                DensityKgM3 = rho,
                Unit = unit,
                EmissionFactorA1A3 = ef,
                ReuseShare = Clamp01(reuse),
                TransportDistanceKm = tKm,
                TransportFactorKgCO2ePerTkm = tEf,
                WasteFactor = Clamp01(wF)
            };

            DA.SetData(0, m);
        }

        static double Clamp01(double x) => x < 0 ? 0 : (x > 1 ? 1 : x);
    }
}
