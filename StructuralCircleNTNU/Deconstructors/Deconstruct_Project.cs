using System;
using Grasshopper.Kernel;
using StructuralCircleNTNU.Classes;

namespace StructuralCircleNTNU.Deconstructors
{
    public class Deconstruct_Project : GH_Component
    {
        public Deconstruct_Project()
          : base("Deconstruct Project", "DePrj",
              "Deconstruct a Project into its components",
              "StructuralCircleNTNU", "Deconstructors")
        { }

        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Project", "P", "Project to deconstruct", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddIntegerParameter("Id", "Id", "Project Id", GH_ParamAccess.item);
            pManager.AddTextParameter("Name", "N", "Project name", GH_ParamAccess.item);
            pManager.AddTextParameter("Location", "Loc", "Project location", GH_ParamAccess.item);
            pManager.AddGenericParameter("SupplyBank", "SB", "Supply Bank", GH_ParamAccess.item);
            pManager.AddGenericParameter("DemandBank", "DB", "Demand Bank", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Project project = null;
            if (!DA.GetData(0, ref project)) return;
            if (project == null) return;

            DA.SetData(0, project.Id);
            DA.SetData(1, project.Name);
            DA.SetData(2, project.Location);
            DA.SetData(3, project.Supply);
            DA.SetData(4, project.Demand);
        }

        protected override System.Drawing.Bitmap Icon => IconLoader.GetIcon();

        public override Guid ComponentGuid => new Guid("A342EFCA-9448-40F4-8834-7B228FF93453");
    }
}
