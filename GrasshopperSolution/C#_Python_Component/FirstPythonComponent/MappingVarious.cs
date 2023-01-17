using System;
using System.Collections.Generic;
using System.Linq;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Grasshopper.Kernel.Data;
using Rhino.Geometry;
using System.IO;
using Serilog;

namespace FirstPythonComponent
{
    public class MappingVarious : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the MappingVarious class.
        /// </summary>
        public MappingVarious()
          : base("MappingVarious", "Nickname",
              "Description",
              "Python", "Mapping")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("ParamNames", "names", "Name of the input parameters", GH_ParamAccess.list); // 0
            pManager.AddGenericParameter("DemandTree", "demand", "Data for the demand tree structure", GH_ParamAccess.tree); // 1
            pManager.AddGenericParameter("SupplyTree", "supply", "Data for the available supply", GH_ParamAccess.tree); // 2
            pManager.AddTextParameter("Method", "method", "Method to use for the substitution", GH_ParamAccess.item, "nestedList");  // 3
            //TODO Create an Enum class or something to simplify this. Maybe together with a value list for the user in Grasshopper. 
            pManager.AddTextParameter("Constraints", "const", "Add constraint dictionary", GH_ParamAccess.list); //4
            pManager[4].Optional = true;

        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Results", "res", "Result from the mapping process", GH_ParamAccess.list); // +
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
            string method = "";
            List<string> constraints = new List<string>(); 

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

            string constraintString = "";
            if (DA.GetDataList(4, constraints))
            {
                foreach (string constraint in constraints)
                {
                    constraintString += (constraint + "\n");
                }

                constraintString = constraintString.Remove(constraintString.Length - 1);
            }

            DA.GetData(3, ref method);

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
            var python_path =
                @"C:\Users\sverremh\OneDrive - NTNU\Autumn 2022\StructuralCircleCode\structuralCircle\MatchingAlgorithms\matching_gh.py";
            var resultInfo = HelperMethods.ExecutePython2(currentDir, python_path, method, inputDemandName, inputSupplyName, resultName, constraintString);
            Log.Information("The Python script has been executed.");



            // -- ouput --
            Log.CloseAndFlush(); // close and clean logger

            // convert back to DataTree
            var outDemand = HelperMethods.NestedListToDataTree(demandLst);
            var results = HelperMethods.DeserializeJson(Path.Combine(fileDir, resultName));




            DA.SetDataList(0, results);

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
                // return Resources.IconForThisComponent;
                return null;
            }
        }

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("61A6DF2C-A3D8-46FD-BCA5-EE87F4837AE0"); }
        }
    }
}