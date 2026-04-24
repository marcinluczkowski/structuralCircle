using System;
using System.Collections.Generic;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using StructuralCircleNTNU.Classes.Sustainability;

namespace StructuralCircleNTNU.Components.Sustainability
{
    /// <summary>
    /// Runs the module-wise LCA on each input DesignConcept for each selected allocation method.
    /// Output is a DataTree with path {concept_idx ; method_idx}.
    /// </summary>
    public class Method_SustainabilityAnalysis : GH_Component
    {
        public Method_SustainabilityAnalysis()
            : base("Sustainability Analysis", "LCA",
                   "Run module-wise LCA (A1–A5, C1–C4, Module D) on a list of DesignConcept using one or more allocation methods. " +
                   "Output is a DataTree {concept ; method}.",
                   "StructuralCircleNTNU", "Sustainability") { }

        public override Guid ComponentGuid => new Guid("E1F2A3B4-0001-4000-8000-000000000004");
        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddGenericParameter("Concepts",  "C",    "DesignConcept list.",                                                      GH_ParamAccess.list);
            p.AddIntegerParameter("Methods",   "M",    "Allocation methods: 0=CutOff, 1=PEF50_50, 2=NCycle. Default = all three.", GH_ParamAccess.list);
            p.AddNumberParameter ("Ncycles",   "Nc",   "Number of cycles (used only by N-cycle allocation).",                      GH_ParamAccess.item, 2.0);
            p.AddNumberParameter ("EoLFactor", "EoL",  "End-of-life C1–C4 factor (kgCO2e per kg of material).",                   GH_ParamAccess.item, 0.03);
            p.AddNumberParameter ("SubEF",     "sEF",  "Substituted material emission factor (kgCO2e per kg) for Module D.",       GH_ParamAccess.item, 0.2);
            p.AddNumberParameter ("DProcess",  "Dp",   "Module D processing CO2e (kgCO2e).",                                      GH_ParamAccess.item, 0.0);
            p.AddBooleanParameter("Run",       "Run",  "Run the analysis. If false the component skips work.",                    GH_ParamAccess.item, true);

            p[1].Optional = true;
        }

        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddGenericParameter("Results", "R",    "SustainabilityResult tree {concept ; method}.", GH_ParamAccess.tree);
            p.AddTextParameter   ("Report",  "Rpt",  "Human-readable report for every branch.",        GH_ParamAccess.tree);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            var conceptRaw = new List<object>();
            var methodIds  = new List<int>();
            double ncycles = 2.0;
            double eol = 0.03;
            double subEF = 0.2;
            double dProc = 0.0;
            bool run = true;

            if (!DA.GetDataList(0, conceptRaw)) return;
            DA.GetDataList(1, methodIds);
            DA.GetData(2, ref ncycles);
            DA.GetData(3, ref eol);
            DA.GetData(4, ref subEF);
            DA.GetData(5, ref dProc);
            DA.GetData(6, ref run);

            if (!run)
            {
                var empty = new DataTree<object>();
                DA.SetDataTree(0, empty);
                DA.SetDataTree(1, empty);
                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "Run is False — analysis skipped.");
                return;
            }

            if (methodIds == null || methodIds.Count == 0)
                methodIds = new List<int> { 0, 1, 2 };

            var resultTree = new DataTree<object>();
            var reportTree = new DataTree<string>();

            var concepts = new List<DesignConcept>();
            foreach (var o in conceptRaw)
            {
                var c = GrasshopperUnpack.AsDesignConcept(o);
                if (c != null) concepts.Add(c);
            }

            if (concepts.Count == 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "No DesignConcept recognised on the Concepts input.");
                return;
            }

            for (int ci = 0; ci < concepts.Count; ci++)
            {
                var concept = concepts[ci];
                for (int mi = 0; mi < methodIds.Count; mi++)
                {
                    int mid = methodIds[mi];
                    if (mid < 0 || mid > 2)
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                            $"Method id {mid} out of range (0–2). Skipping.");
                        continue;
                    }

                    var method = (AllocationMethod)mid;
                    var result = SustainabilityEngine.CalculateConcept(
                        concept, method, ncycles, eol, subEF, dProc);

                    var path = new GH_Path(ci, mi);
                    resultTree.Add(result, path);
                    reportTree.Add(result.ToString(), path);
                }
            }

            DA.SetDataTree(0, resultTree);
            DA.SetDataTree(1, reportTree);
        }
    }
}
