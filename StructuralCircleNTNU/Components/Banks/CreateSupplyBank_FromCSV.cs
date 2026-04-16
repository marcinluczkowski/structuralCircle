using System;
using System.Collections.Generic;
using System.IO;
using Grasshopper.Kernel;
using Rhino.Geometry;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Components.Banks
{
    public class CreateSupplyBank_FromCSV : GH_Component
    {
        public CreateSupplyBank_FromCSV()
          : base("Supply Bank (CSV)", "SBankCSV",
              "Create a Supply Bank by reading beam/plate data from a CSV file",
              "StructuralCircleNTNU", "Banks")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("FilePath", "F", "Path to CSV file", GH_ParamAccess.item);
            pManager.AddTextParameter("Type", "T", "Element type: 'Beam' or 'Plate'", GH_ParamAccess.item, "Beam");
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("SupplyBank", "SB", "Constructed Supply Bank from CSV", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            string filePath = "";
            string type = "Beam";
            DA.GetData(0, ref filePath);
            DA.GetData(1, ref type);

            if (!File.Exists(filePath))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"File not found: {filePath}");
                return;
            }

            var elements = CsvParser.ParseElements(filePath, type, this);
            if (elements == null) return;

            var bank = new SupplyBank(elements);
            DA.SetData(0, bank);
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("A1B2C3D4-1111-4000-8000-000000000012");
    }
}
