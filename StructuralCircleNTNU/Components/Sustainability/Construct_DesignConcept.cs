using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes.Sustainability;

namespace StructuralCircleNTNU.Components.Sustainability
{
    public class Construct_DesignConcept : GH_Component
    {
        public Construct_DesignConcept()
            : base("Design Concept", "Concept",
                   "Combine material items, connection items and service-life parameters into a DesignConcept used by the sustainability analysis.",
                   "StructuralCircleNTNU", "Sustainability") { }

        public override Guid ComponentGuid => new Guid("E1F2A3B4-0001-4000-8000-000000000003");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddTextParameter   ("Name",          "N",    "Concept name.",                                          GH_ParamAccess.item, "Concept");
            p.AddGenericParameter("Materials",     "Mats", "List of MaterialItem.",                                  GH_ParamAccess.list);
            p.AddGenericParameter("Connections",   "Conn", "List of ConnectionItem.",                                GH_ParamAccess.list);
            p.AddNumberParameter ("StudyPeriod",   "Yrs",  "Reference study period (years).",                        GH_ParamAccess.item, 50.0);
            p.AddNumberParameter ("AreaM2",        "m²",   "Functional unit area (m²).",                             GH_ParamAccess.item, 1.0);
            p.AddNumberParameter ("FutureReuseProb","FR",  "Probability that materials are reused after use (0–1).", GH_ParamAccess.item, 0.5);
            p.AddNumberParameter ("SubstitutionF", "SF",   "Substitution factor for Module D (0–1).",                GH_ParamAccess.item, 1.0);
            p.AddNumberParameter ("UtilRatio",     "U",    "Average structural utilisation ratio (0–1).",            GH_ParamAccess.item, 1.0);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddGenericParameter("DesignConcept", "Concept", "Design concept.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            string name = "Concept";
            var matRaw = new List<object>();
            var connRaw = new List<object>();
            double years = 50, area = 1, frp = 0.5, sf = 1.0, util = 1.0;

            DA.GetData(0, ref name);
            DA.GetDataList(1, matRaw);
            DA.GetDataList(2, connRaw);
            DA.GetData(3, ref years);
            DA.GetData(4, ref area);
            DA.GetData(5, ref frp);
            DA.GetData(6, ref sf);
            DA.GetData(7, ref util);

            var materials = new List<MaterialItem>();
            foreach (var o in matRaw)
            {
                var m = GrasshopperUnpack.AsMaterialItem(o);
                if (m != null) materials.Add(m);
            }

            var connections = new List<ConnectionItem>();
            foreach (var o in connRaw)
            {
                var c = GrasshopperUnpack.AsConnectionItem(o);
                if (c != null) connections.Add(c);
            }

            if (materials.Count == 0 && matRaw.Count > 0)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No MaterialItem recognised in the Materials input.");
            if (connections.Count == 0 && connRaw.Count > 0)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No ConnectionItem recognised in the Connections input.");

            var concept = new DesignConcept
            {
                Name = name,
                Materials = materials,
                Connections = connections,
                ReferenceStudyPeriodYears = years,
                FunctionalUnitAreaM2 = area,
                FutureReuseProbability = Clamp01(frp),
                SubstitutionFactor = Clamp01(sf),
                UtilisationRatio = Clamp01(util)
            };

            DA.SetData(0, concept);
        }

        static double Clamp01(double x) => x < 0 ? 0 : (x > 1 ? 1 : x);
    }
}
