using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes.Sustainability;

namespace StructuralCircleNTNU.Deconstructors
{
    public class Deconstruct_DesignConcept : GH_Component
    {
        public Deconstruct_DesignConcept()
            : base("Deconstruct Design Concept", "DeconConcept",
                   "Explode a DesignConcept into its fields.",
                   "StructuralCircleNTNU", "Deconstructors") { }

        public override Guid ComponentGuid => new Guid("E1F2A3B4-0001-4000-8000-000000000006");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddGenericParameter("Concept", "C", "DesignConcept to deconstruct.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddTextParameter   ("Name",        "N",     "Concept name.",                                GH_ParamAccess.item);
            p.AddGenericParameter("Materials",   "Mats",  "Material items.",                              GH_ParamAccess.list);
            p.AddGenericParameter("Connections", "Conn",  "Connection items.",                            GH_ParamAccess.list);
            p.AddNumberParameter ("StudyPeriod", "Yrs",   "Reference study period (years).",              GH_ParamAccess.item);
            p.AddNumberParameter ("Area",        "m²",    "Functional unit area (m²).",                   GH_ParamAccess.item);
            p.AddNumberParameter ("FutureReuseProb","FR", "Probability of future reuse (0–1).",           GH_ParamAccess.item);
            p.AddNumberParameter ("SubstitutionF","SF",   "Substitution factor for Module D (0–1).",      GH_ParamAccess.item);
            p.AddNumberParameter ("UtilRatio",   "U",     "Average structural utilisation ratio (0–1).", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            object raw = null;
            if (!DA.GetData(0, ref raw)) return;

            var c = GrasshopperUnpack.AsDesignConcept(raw);
            if (c == null)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input is not a DesignConcept.");
                return;
            }

            DA.SetData     (0, c.Name);
            DA.SetDataList (1, c.Materials);
            DA.SetDataList (2, c.Connections);
            DA.SetData     (3, c.ReferenceStudyPeriodYears);
            DA.SetData     (4, c.FunctionalUnitAreaM2);
            DA.SetData     (5, c.FutureReuseProbability);
            DA.SetData     (6, c.SubstitutionFactor);
            DA.SetData     (7, c.UtilisationRatio);
        }
    }
}
