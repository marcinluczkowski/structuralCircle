using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes.Sustainability;

namespace StructuralCircleNTNU.Components.Sustainability
{
    public class Construct_ConnectionItem : GH_Component
    {
        public Construct_ConnectionItem()
            : base("Connection Item", "ConnItem",
                   "Create one connection line (steel, interlocking timber, or hybrid) for a DesignConcept.",
                   "StructuralCircleNTNU", "Sustainability") { }

        public override Guid ComponentGuid => new Guid("E1F2A3B4-0001-4000-8000-000000000002");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddTextParameter  ("Name",           "N",    "Connection label.",                               GH_ParamAccess.item, "");
            p.AddIntegerParameter("Count",         "#",    "Number of this connection type.",                 GH_ParamAccess.item, 1);

            p.AddNumberParameter("SteelMassKg",    "St_kg","Steel mass per single connection (kg).",          GH_ParamAccess.item, 0.0);
            p.AddNumberParameter("SteelEF",        "St_EF","Steel emission factor (kgCO2e per kg).",          GH_ParamAccess.item, 1.5);

            p.AddNumberParameter("ExtraTimberKg",  "T_kg", "Extra timber per interlocking connection (kg).",  GH_ParamAccess.item, 0.0);
            p.AddNumberParameter("TimberEF",       "T_EF", "Timber emission factor (kgCO2e per kg).",         GH_ParamAccess.item, 0.35);

            p.AddNumberParameter("CncEnergyKWh",   "kWh",  "CNC energy per connection (kWh).",                GH_ParamAccess.item, 0.0);
            p.AddNumberParameter("ElecEF",         "E_EF", "Electricity factor (kgCO2e per kWh).",            GH_ParamAccess.item, 0.1);

            p.AddNumberParameter("ToolWearCO2e",   "Tw",   "Tool-wear CO2e per connection (kgCO2e).",         GH_ParamAccess.item, 0.0);
            p.AddNumberParameter("YieldLoss",      "YL",   "CNC yield loss fraction (0–1).",                  GH_ParamAccess.item, 0.0);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddGenericParameter("ConnectionItem", "C", "Connection item.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            string name = "";
            int count = 1;
            double stKg = 0, stEf = 1.5, tKg = 0, tEf = 0.35, kwh = 0, eEf = 0.1, tw = 0, yl = 0;

            DA.GetData(0, ref name);
            DA.GetData(1, ref count);
            DA.GetData(2, ref stKg);  DA.GetData(3, ref stEf);
            DA.GetData(4, ref tKg);   DA.GetData(5, ref tEf);
            DA.GetData(6, ref kwh);   DA.GetData(7, ref eEf);
            DA.GetData(8, ref tw);    DA.GetData(9, ref yl);

            var c = new ConnectionItem
            {
                Name = name,
                Count = Math.Max(0, count),
                SteelMassKg = stKg,
                SteelEmissionFactor = stEf,
                ExtraTimberKg = tKg,
                TimberEmissionFactor = tEf,
                CncEnergyKWh = kwh,
                ElectricityEmissionFactor = eEf,
                ToolWearCO2e = tw,
                YieldLoss = Math.Max(0, Math.Min(1, yl))
            };

            DA.SetData(0, c);
        }
    }
}
