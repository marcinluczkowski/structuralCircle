using System;
using System.Collections.Generic;
using Grasshopper.Kernel.Parameters;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System.Runtime.Versioning;
using Grasshopper;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using MatchingWrapper;
using FirstPythonComponent;
using System.Reflection;
using System.IO;

namespace MatchingWrapper
{
    public class BetaWrapper : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the BetaWrapper class.
        /// </summary>
        public BetaWrapper()
          : base("BetaWrapper", "Nickname",
              "Description",
              "Python", "Reuse")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("DemandData", "demand", "DataTree with demand elements. First branch should be column names", GH_ParamAccess.tree); // 0
            pManager.AddGenericParameter("SupplyData", "supply", "DataTree with demand elements. First branch should be column names", GH_ParamAccess.tree); // 1
            pManager.AddIntegerParameter("Method", "method", "Select the method to be used. Right click to see options", GH_ParamAccess.item, 1); //2
            pManager.AddTextParameter("MatchingConstraints", "constraints", "Specify constraints to use during mapping", GH_ParamAccess.list); // 3
            pManager.AddTextParameter("FixedElements", "fixEls", "", GH_ParamAccess.list); //4

            Param_Integer methodParam = pManager[2] as Param_Integer;
            methodParam.AddNamedValue("GreedyS", 0);
            methodParam.AddNamedValue("GreedyP", 1);
            methodParam.AddNamedValue("Bipartite", 2);
            methodParam.AddNamedValue("MIP", 3);


            pManager[4].Optional = true;
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddIntegerParameter("Matching", "matching", "Results from the matching", GH_ParamAccess.tree); // 0
            pManager.AddTextParameter("Result info", "info", "Info about the matching process", GH_ParamAccess.list); // 1
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object is used to retrieve from inputs and store in outputs.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            // -- input --
            GH_Structure<IGH_Goo> demandTree; // 0
            GH_Structure<IGH_Goo> supplyTree; // 1

            int methodInt = 0; // 2
            List<string> constraints = new List<string>(); //3
            List<string> lockedPositions = new List<string>(); // 4

            if (!DA.GetDataTree(0, out demandTree)) return; // 0
            if (!DA.GetDataTree(1, out supplyTree)) return; // 1
            DA.GetData(2, ref methodInt); // 2
            if (!DA.GetDataList(3, constraints)) // 3
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "There need to be some constraints for the matching");
                return;
            }
            DA.GetDataList(4, lockedPositions); // 4


            // -- method --

            // convert trees to nested lists
            List<List<object>> demandList = HelperMethods.TreeToNestedList(demandTree);
            List<List<object>> supplyList = HelperMethods.TreeToNestedList(supplyTree);

            // set the file path for demand and supply
            string projectName = Assembly.GetCallingAssembly().GetName().Name;
            string username = Environment.UserName;
            string existingDir = String.Format(@"C:\Users\{0}\AppData\Roaming\Grasshopper\Libraries\MatchingWrapper", username);
            var fileDir = Path.Combine(existingDir, "PythonFiles"); // look for a folder with TempFiles. If not found, create it. 
            if (!Directory.Exists(fileDir))
            {
                var dir = Directory.CreateDirectory(fileDir);                
            }
            Directory.SetCurrentDirectory(fileDir);
            
            string demandPath = "demand.json";
            string supplyPath = "supply.json";
            

            // write the data to Json format
            HelperMethods.JsonFromList(demandList, fileDir, demandPath);
            HelperMethods.JsonFromList(supplyList, fileDir, supplyPath);

            // Exectute matching.py script from terminal. This will probably only work as long as the user has all the methods available. Need to build a package?


            string status = HelperMethods.ExecutePython3();
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, status); // add the status about results
            

            // read the csv file created in batch/python






            // -- results --



            // -- output --

        }

        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                //You can add image files to your project resources and access them like this:
                return Properties.Resources.beta_wrap;
                

            }
        }

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("093A9A6D-4343-4838-9A65-DBF5D13940AB"); }
        }
    }
}