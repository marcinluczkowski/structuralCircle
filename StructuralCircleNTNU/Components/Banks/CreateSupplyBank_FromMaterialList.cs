using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Banks
{
    public class CreateSupplyBank_FromMaterialList : GH_Component
    {
        public CreateSupplyBank_FromMaterialList()
          : base("Supply Bank (Material List)", "SBankML",
              "Create a Supply Bank from a material-list CSV with columns:\n" +
              "  MaterialType, Width, Height, Length, Quantity (all lengths in the chosen Unit, converted to m)\n" +
              "Element type (Beam / Plate) is inferred automatically:\n" +
              "  'Limtre', 'GL*'         → Beam\n" +
              "  'X-LAM', 'CLT', wide-W  → Plate\n" +
              "Material and Section instances are created automatically.",
              "StructuralCircleNTNU", "Banks")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("FilePath", "F", "Path to material list CSV file", GH_ParamAccess.item);
            pManager.AddTextParameter("Unit", "U", "Length unit of Width, Height, Length in CSV: mm, cm, or m (default mm).", GH_ParamAccess.item, "mm");
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("SupplyBank",  "SB",  "Supply Bank built from CSV",          GH_ParamAccess.item);
            pManager.AddGenericParameter("Elements",    "E",   "All individual elements (expanded)",   GH_ParamAccess.list);
            pManager.AddGenericParameter("Beams",       "B",   "Beam elements only",                   GH_ParamAccess.list);
            pManager.AddGenericParameter("Plates",      "P",   "Plate elements only",                  GH_ParamAccess.list);
            pManager.AddTextParameter(  "Report",       "Rpt", "Summary report",                       GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            string filePath = "";
            string unit = "mm";
            if (!DA.GetData(0, ref filePath)) return;
            DA.GetData(1, ref unit);

            if (!File.Exists(filePath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"File not found: {filePath}");
                return;
            }

            var elements = CsvParser.ParseMaterialList(filePath, this, unit);
            if (elements == null) return;

            var bank   = new SupplyBank(elements);
            var beams  = bank.Beams.Cast<Element>().ToList();
            var plates = bank.Plates.Cast<Element>().ToList();

            var report = BuildReport(bank, filePath);

            DA.SetData(0, bank);
            DA.SetDataList(1, elements);
            DA.SetDataList(2, beams);
            DA.SetDataList(3, plates);
            DA.SetData(4, report);
        }

        static string BuildReport(SupplyBank bank, string filePath)
        {
            var beams  = bank.Beams;
            var plates = bank.Plates;

            var matGroups = bank.Elements
                .GroupBy(e => e.Material?.Name ?? "Unknown")
                .OrderBy(g => g.Key);

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"Supply Bank from: {Path.GetFileName(filePath)}");
            sb.AppendLine($"Total elements : {bank.Count}  ({beams.Count} beams, {plates.Count} plates)");
            sb.AppendLine();
            sb.AppendLine("── By material type ──────────────────────────");
            foreach (var grp in matGroups)
                sb.AppendLine($"  {grp.Key,-30} {grp.Count(),4} elements");
            sb.AppendLine();
            sb.AppendLine("── Beam sections ─────────────────────────────");
            foreach (var grp in beams
                .GroupBy(b => $"{b.Section?.Name ?? "?"}")
                .OrderBy(g => g.Key))
                sb.AppendLine($"  {grp.Key,-20} {grp.Count(),4} pcs  L={string.Join(", ", grp.Select(b => $"{b.Length:F3}m"))}");
            sb.AppendLine();
            sb.AppendLine("── Plate sections ────────────────────────────");
            foreach (var grp in plates
                .GroupBy(p => $"{p.Section?.Name ?? "?"}")
                .OrderBy(g => g.Key))
                sb.AppendLine($"  {grp.Key,-20} {grp.Count(),4} pcs");
            return sb.ToString();
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("A1B2C3D4-1111-4000-8000-000000000020");
    }
}
