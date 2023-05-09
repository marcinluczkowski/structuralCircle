using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using GH_IO.Types;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using Serilog;

using Serilog.Core;

namespace FirstPythonComponent
{
    public class MappingGraph : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the LineMidpoint class.
        /// </summary>
        public MappingGraph()
          : base("MappingGraph", "Graph_map",
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
            pManager.AddGenericParameter("Results", "res", "Result from the mapping process", GH_ParamAccess.tree); 
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="da">The DA object is used to retrieve from inputs and store in outputs.</param>
        protected override void SolveInstance(IGH_DataAccess da)
        {
            // -- input -- 
            List<string> paramNames = new List<string>();
            GH_Structure<IGH_Goo> demand; 
            GH_Structure<IGH_Goo> supply;

            da.GetDataList(0, paramNames);
            
            if (!da.GetDataTree(1, out demand)) return;
            if (!da.GetDataTree(2, out supply)) return;

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
            string pythonName = "mapping_graph.py";

            // create logger
            var logPath = fileDir + "\\log.txt";

            // create logger
            File.Delete(logPath);
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Debug()
                .WriteTo.File(logPath)
                .CreateLogger();
            Log.Information("The global logger has been configured");

            var watch = new Stopwatch();// Initiate a stopwatch


            // deserialize json and write to file
            JsonFromList(demandLst, fileDir, inputDemandName);
            JsonFromList(supplyLst, fileDir, inputSupplyName);


            // execute Python script from console
            var resultInfo = ExecutePython(currentDir, pythonName, inputDemandName, inputSupplyName, resultName);

            // get result as nested list
            var resultList = HelperMethods.CsvToNestedList(Path.Combine(fileDir,resultName));
            
            

            // -- ouput --
            Log.CloseAndFlush(); // close and clean logger
            
            // convert back to DataTree
            var outDemand = HelperMethods.NestedListToDataTree(demandLst);
            var outResult = HelperMethods.NestedListToDataTree(resultList.Select(lst => lst.Select(el => el as object).ToList()).ToList());
            da.SetDataTree(0, outResult);
            //DA.SetData(1, logString);

        }

        #region Methods

        private void JsonFromList(List<List<object>> nestedList, string path, string filename)
        {
            var json = JsonSerializer.Serialize(nestedList);
            File.WriteAllText(Path.Combine(path, filename) ,json);
        }
        
        private void CreateJsonFromDict(Dictionary<string, List<object>> dict, string dir, string name)
        { 
            //var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(dict);
            File.WriteAllText(Path.Combine(dir, name), json);

        }
        private Tuple<string, string> ExecutePython(string filePath, string pythonFile, string demandName, string supplyName, string resultName)
        {
            // 1) Create Process Info
            var psi = new ProcessStartInfo();
            var username = Environment.UserName;
            psi.FileName = String.Format(@"C:\Users\{0}\AppData\Local\Programs\Python\Python39\python.exe", username);
            // 2) Provide script and arguments
            var script = Path.Combine(filePath, "PythonFiles", pythonFile);
            var demandJson = Path.Combine(filePath, "Files", demandName);
            var supplyJson = Path.Combine(filePath, "Files", supplyName);
            var outputJson = Path.Combine(filePath, "Files", resultName);
            var args = new List<string>(){script, demandJson, supplyJson, outputJson};
            var argStr = $"";
            foreach (var arg in args)
            {
                argStr += ("\"" + arg + "\" ");
            }
            psi.Arguments = argStr;

            // 3) Process configuration
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardError = true;
            psi.RedirectStandardOutput = true;

            // 4) Execute process and get output
            var errors = "";
            var results = "";

            using (var process = Process.Start(psi))
            {
                try
                {
                    var w = new System.Diagnostics.Stopwatch();
                    Log.Information("Reading results from shell:"); w.Start();
                    errors = process.StandardError.ReadToEnd();
                    results = process.StandardOutput.ReadToEnd();
                    Log.Debug("Time used to read results from console: {time} ms", w.ElapsedMilliseconds); w.Reset();
                }
                catch (Exception e)
                {

                }
            }
            return new Tuple<string, string>(errors, results);
        }

        private Dictionary<string, List<string>> DeserializeJson(string path)
        {
            var dict = new Dictionary<string, List<string>>();
            var json = File.ReadAllText(path);
            var vals = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json);
            foreach (var kvp in vals)
            {
                var val = vals[kvp.Key].ToString();
                var strList = val.Split(',').Select(x => x.Split(':')[1]).ToList();
                strList[strList.Count - 1] = Regex.Replace(strList[strList.Count - 1], "[}]", string.Empty);
                dict.Add(kvp.Key, strList);
            }

            return dict;
        }
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
            get { return new Guid("8bda2dec-76a2-46ec-80e9-4474ec230cfb"); }
        }
    }
}