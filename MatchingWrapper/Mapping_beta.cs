using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.AccessControl;
using System.Text.Json;
using System.Text.RegularExpressions;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using Serilog;
using Serilog.Core;

namespace MatchingWrapper
{
    public class MappingBeta : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the LineMidpoint class.
        /// </summary>
        public MappingBeta()
          : base("MappingBeta", "map",
              "Deserializes input tree and runs the mapping algorithm through a python script before returning the output.",
              "Python", "Mapping")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("ParamNames", "names", "Name of the input parameters", GH_ParamAccess.list);
            pManager.AddGenericParameter("DemandTree", "demand", "Data for the demand tree structure", GH_ParamAccess.tree);
            pManager.AddGenericParameter("SupplyTree", "supply", "Data for the available supply", GH_ParamAccess.tree);

            pManager[0].Optional = true;
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Results", "res", "Result from the mapping process", GH_ParamAccess.list); 
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object is used to retrieve from inputs and store in outputs.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            // -- input -- 
            List<string> paramNames = new List<string>();
            GH_Structure<IGH_Goo> demand; 
            GH_Structure<IGH_Goo> supply;

            DA.GetDataList(0, paramNames);
            
            if (!DA.GetDataTree(1, out demand)) return;
            if (!DA.GetDataTree(2, out supply)) return;

            if (paramNames.Count > 0 && paramNames.Count != demand.get_Branch(0).Count)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "The length of param names does not match the number of params and will be ignored");
                paramNames = Enumerable.Range(0, demand.get_Branch(0).Count).Select(el => el.ToString()).ToList(); // use numbers as param headers / column names
            }
            else if (paramNames.Count == 0)
            {
                paramNames = Enumerable.Range(0, demand.get_Branch(0).Count).Select(el => el.ToString()).ToList(); // use numbers as param headers / column names
            }

            // -- method -- 
            
            // convert to NestedList
            var demandLst = HelperMethods.TreeToNestedList(demand);
            var supplyLst = HelperMethods.TreeToNestedList(supply);
            
            demandLst.Insert(0, paramNames.Select(el => el as object).ToList());
            supplyLst.Insert(0, paramNames.Select(el => el as object).ToList());
            
            //  -- directory and names
            //var GHFolder = Folders.AppDataFolder;

            //var currentDir = Path.Combine(GHFolder, @"Libraries\FirstPython\Debug\net48");
            var username = Environment.UserName;
            var currentDir = String.Format(@"C:\Users\{0}\AppData\Roaming\Grasshopper\Libraries\FirstPython\Debug\net48", username);
            var fileDir = Path.Combine(currentDir, "Files");
            if (!Directory.Exists(fileDir))
            {
                var dir = Directory.CreateDirectory(fileDir);
                Directory.SetCurrentDirectory(fileDir);
            }
            string inputDemandName = "demand_input.json"; string inputSupplyName = "supply_input.json"; string resultName = "output.json";
            string pythonName = "mapping_beta.py";
            
            // create logger
            var logPath = fileDir + "\\log.txt";
            
            
            File.Delete(logPath);
            
            
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Debug()
                .WriteTo.File(logPath)
                .CreateLogger();
            Log.Information("The global logger has been configured");
            
            var watch = new System.Diagnostics.Stopwatch();// Initiate a stopwatch



            // deserialize json and write to file
            HelperMethods.JsonFromList(demandLst, fileDir, inputDemandName);
            HelperMethods.JsonFromList(supplyLst, fileDir, inputSupplyName);
            Log.Information("The input has been deserialized and written to json files.");

            // execute Python script from console
            

            var resultInfo = HelperMethods.ExecutePython(currentDir, pythonName, inputDemandName, inputSupplyName, resultName);
            Log.Information("The Python script has been executed.");

            

            // -- ouput --
            Log.CloseAndFlush(); // close and clean logger
            
            // convert back to DataTree
            var outDemand = HelperMethods.NestedListToDataTree(demandLst);
            var results = HelperMethods.DeserializeJson(Path.Combine(fileDir, resultName));
            

            

            DA.SetDataList(0, results);
            

        }

        #region Methods

       
        #endregion


        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                //You can add image files to your project resources and access them like this:
                // return Resources.IconForThisComponent;
                return null;
            }
        }

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("0745042e-b882-45b1-913a-b0a930612e46"); }
        }
    }
}