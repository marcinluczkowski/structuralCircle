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
using System.Reflection;
using System.IO;
using System.Linq;
using System.Text.Json;
using Grasshopper.Kernel.Special;
using System.ComponentModel;

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
            pManager.AddTextParameter("Parameter Names", "names", "List of names for each parameter used in the data trees", GH_ParamAccess.list); // 2
            pManager.AddIntegerParameter("Method", "method", "Select the method to be used. Right click to see options", GH_ParamAccess.item); //3
            pManager.AddTextParameter("MatchingConstraints", "constraints", "Specify constraints to use during mapping", GH_ParamAccess.item); // 4
            pManager.AddTextParameter("FixedElements", "fixEls", "", GH_ParamAccess.list); //5


            pManager[5].Optional = true;
            Param_Integer methodParam = pManager[3] as Param_Integer;
            methodParam.AddNamedValue("GreedyS", 0);
            methodParam.AddNamedValue("GreedyP", 1);
            methodParam.AddNamedValue("Bipartite", 2);
            methodParam.AddNamedValue("MIP", 3);            

        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Matching", "matching", "Results from the matching", GH_ParamAccess.list); // 0
            pManager.AddTextParameter("Result info", "info", "Info about the matching process", GH_ParamAccess.list); // 1
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object is used to retrieve from inputs and store in outputs.</param>
        
        

        protected override void BeforeSolveInstance()
        {
            // Before running the component I add the value lists I need
            var method_dict = new Dictionary<string, string>() { { "Greedy Single", "0" }, { "Greedy Plural", "1" }, { "MaxBM", "2" }, { "MILP", "3" } };
            CreateValueListString(this, method_dict, 3);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            
            


            // -- input --
            GH_Structure<IGH_Goo> demandTree; // 0
            GH_Structure<IGH_Goo> supplyTree; // 1
            List<string> names = new List<string>(); // 2
            int methodInt = 0; // 3
            string constraints = ""; //4
            List<string> lockedPositions = new List<string>(); // 5

            if(!DA.GetDataTree(0, out demandTree))return;// 0
            if (!DA.GetDataTree(1, out supplyTree)) return; // 1
            if(DA.GetDataList(2, names))return; // 1
            DA.GetData(3, ref methodInt); // 3
            if (!DA.GetData(4, ref constraints)) // 4
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "There need to be some constraints for the matching");
            }
            DA.GetDataList(5, lockedPositions); // 5


            

            // -- method --

            // convert trees to nested lists
            List<List<object>> demandList = HelperMethods.TreeToNestedList(demandTree);
            List<List<object>> supplyList = HelperMethods.TreeToNestedList(supplyTree);

            // ensure that all demandList has the same number of entries, and that the number of column names match the number of properties
            List<int> uniqueDemandLstCounts = demandList.Select(lst => lst.Count).Distinct().ToList(); // number of entries in each sub list
            List<int> uniqueSupplyLstCounts = supplyList.Select(lst => lst.Count).Distinct().ToList(); // number of entries in each sub list
            if (uniqueDemandLstCounts.Count > 1 || uniqueSupplyLstCounts.Count > 1)
            {
                // If there are more than one unique length of all sublist or the the unique length does not match the number of names. throw a warning
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "The number of attributes are not equal for all elements. Please check the input data");
            }
            else if (uniqueDemandLstCounts[0] != names.Count || uniqueSupplyLstCounts[0] != names.Count)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "There need to be a name for each attribute assigned to demand and supply elements.");
                return;
            }
            else
            {
                // Input data should be good. Add names to the lists and proceed.
                demandList.Insert(0, names.Select(el => el as object).ToList());
                supplyList.Insert(0, names.Select(el => el as object).ToList());
            }
            
            


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


            var resultString = HelperMethods.ExecuteBatch(methodNumber: methodInt, demandPath: Path.Combine(fileDir, demandPath), supplyPath: Path.Combine(fileDir, supplyPath), constraints: constraints);

            // Read the result from result.py
            List<string> matching_pairs = new List<string>();
            try
            {
                string resultText = File.ReadAllText(Path.Combine(fileDir, "result.json"));
                var matchingResults = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string,string>>>(resultText);
                var pairDict = matchingResults["Supply_id"];
                foreach (KeyValuePair<string, string> entry in pairDict) {
                    matching_pairs.Add(entry.Key + ":" + entry.Value);
                }

            }
            catch (Exception)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Could not find result file and deserialize to dictionary.");
            }
            


            /*
            // read the csv file created in batch/python
            string csvPath = Path.Combine(fileDir, "first_test.csv");
            List<string> listA = new List<string>(); // empty list
            List<string> listB = new List<string>(); // empty list
            List<string> csvItems = new List<string>();

            using (var reader = new StreamReader(csvPath))
            {
                while (!reader.EndOfStream)
                {
                    string line = reader.ReadLine();
                    string[] values = line.Split(',');
                    csvItems.Add(line);
                    listA.Add(values[0]);
                    listB.Add(values[1]);
                }

                listA = listA.Where(el => el != "").ToList();
                listA = listB.Where(el => el != "").ToList();
            }

            */


            // -- results --



            // -- output --
            DA.SetDataList(0, matching_pairs);
            DA.SetDataList(1, String.Join("\n", resultString));
        }
        /// <summary>
        /// Creates a ValueList for strings and connects it to the selected input.
        /// </summary>
        /// <param name="component">This Grasshopper component.</param>
        /// <param name="valueListDict">A dictionary with the key used as the name, and value as the value in the ValueList</param>
        /// <param name="inputNum">The index of the input to connect the ValueSlider to.</param>
        private void CreateValueListString(GH_Component component, Dictionary<string, string> valueListDict, int inputNum)
        {
            //instantiate  new value list
            var valueList = new Grasshopper.Kernel.Special.GH_ValueList(); // instantiate the new value list
            valueList.CreateAttributes();


            int inputCount = Params.Input[inputNum].SourceCount; // the input to connect to
            if (inputCount != 0) { return; } // We do not want to create a new slider if it already exists.
            //set the position of the ValueList in the canvas
            valueList.Attributes.Pivot = new System.Drawing.PointF((float)Attributes.Pivot.X - Attributes.Bounds.Width - valueList.Attributes.Bounds.Width/2 - 3,
                (float)Attributes.Pivot.Y);

            // populate value list with data from input dictionary
            valueList.ListItems.Clear(); // clear the initial values
            foreach (KeyValuePair<string, string> kvp in valueListDict)
            {
                var listItem = new GH_ValueListItem(kvp.Key, kvp.Value);
                valueList.ListItems.Add(listItem); // add item to valueList
            }

            // Add ValueList to canvas
            Grasshopper.Instances.ActiveCanvas.Document.AddObject(valueList, false); // set recompute to "true" to add the value slider before starting the rest of the MatchingWrapper

            // connect slider to this component
            Params.Input[inputNum].AddSource(valueList);




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