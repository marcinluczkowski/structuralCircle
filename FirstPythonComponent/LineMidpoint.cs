using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using Grasshopper.Kernel;
using Rhino.Geometry;
using Serilog;

using Serilog.Core;

namespace FirstPythonComponent
{
    public class LineMidpoint : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the LineMidpoint class.
        /// </summary>
        public LineMidpoint()
          : base("LineMidpoint", "Nickname",
              "Description",
              "Python", "Primitive")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddCurveParameter("Curves", "crvs", "List of curves", GH_ParamAccess.list);
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddPointParameter("Midpoints", "pts", "Midpoints of input curves", GH_ParamAccess.list); // 0
            pManager.AddTextParameter("Logger", "log", "Solution information", GH_ParamAccess.item); // 1
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="da">The DA object is used to retrieve from inputs and store in outputs.</param>
        protected override void SolveInstance(IGH_DataAccess da)
        {
            // -- input -- 
            List<Curve> lines = new List<Curve>();

            if (!da.GetDataList(0, lines)) return;

            // -- method -- 

            // directory and names
            string fileDirStr = "Files";  string inputName = "input.json";  string resultName = "output.json";
            string baseDir = Directory.GetCurrentDirectory();
            string fileDir = Path.Combine(baseDir, fileDirStr);
            Directory.CreateDirectory(fileDir);
            //Directory.SetCurrentDirectory(fileDir);
            string testDir = Directory.GetCurrentDirectory();
            
            // create logger
            var logPath = Path.Combine(fileDir, "log.txt");
            File.Delete(logPath);
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Debug()
                .WriteTo.File(logPath)
                .CreateLogger();
            Log.Information("The global logger has been configured");
            var watch = new System.Diagnostics.Stopwatch();// Initiate a stopwatch

            Log.Information("Creating dictionary for endpoints"); watch.Start();
            var endPointsDict = GetEndPointDict(lines);
            watch.Stop();
            Log.Debug("Elapsed time: {time} ms", watch.ElapsedMilliseconds); watch.Reset();


            // create json file from dict
            Log.Information("Creating json from dictionary:"); watch.Start();
            //var filepath = Path.Combine(dir, "endpoint.json");
            //var jsonPath = @"C:\Users\sverremh\source\repos\FirstPythonComponent\FirstPythonComponent\endpoints.json"; // generalize this later
            
            CreateJsonFromDict(endPointsDict, fileDir, inputName);
            watch.Stop(); 
            Log.Debug("Elapsed time: {time} ms", watch.ElapsedMilliseconds); watch.Reset();


            // Execute Python code
            Log.Information("Executing the python code from command prompt:"); watch.Start();
            //string scriptPath = @"C:\Users\sverremh\source\repos\FirstPythonComponent\FirstPythonComponent\python_json.py";

            string pythonFile = "python_json.py";
            var results = ExecutePython(baseDir, pythonFile, inputName, resultName);
            Log.Debug("Elapsed time: {time} ms", watch.ElapsedMilliseconds); watch.Reset();

            // Deserialize JSON and input into dictionary
            Log.Information("Deserializing JSON and writing to result dict:"); watch.Start();
            Dictionary<string, List<string>> resultDict;
            List<Point3d?> midPts = new List<Point3d?>();
            if (results.Item1 == "")
            {
                resultDict = DeserializeJson(Path.Combine(fileDir, resultName));
                midPts = PointsFromDict(resultDict);
            }
            else
            {
                Log.Error("Result could not be retrieved.");
            }
            Log.Debug("Elapsed time: {time} ms", watch.ElapsedMilliseconds); watch.Reset();

            // -- ouput --
            string logString = Log.Logger.ToString(); // create a string from log
            Log.CloseAndFlush(); // close and clean logger
            //Directory.SetCurrentDirectory(baseDir); // reset directory
            
            
            da.SetDataList(0, midPts);
            da.SetData(1, logString);

        }

        #region Methods

        private List<Point3d?> PointsFromDict(Dictionary<string, List<string>> dict)
        {
            var midPts = new List<Point3d?>();

            var keys = dict.Keys.ToList();
            int count = dict[keys[0]].Count;

            for (int i = 0; i < count; i++)
            {
                bool isx = float.TryParse(dict[keys[0]][i], out float x);
                bool isy = float.TryParse(dict[keys[1]][i], out float y);
                bool isz = float.TryParse(dict[keys[2]][i], out float z);

                if (isx && isy && isx)
                {
                    midPts.Add(new Point3d(x,y,z));
                }
                else
                {
                    midPts.Add(null);
                }
            }
            

            return midPts;
        }

        private Dictionary<string, List<object>> GetEndPointDict(List<Curve> curves)
        {
            var dict = new Dictionary<string, List<object>>()
            {
                {"X0", new List<object>()},
                {"Y0", new List<object>()},
                {"Z0", new List<object>()},
                {"X1", new List<object>()},
                {"Y1", new List<object>()},
                {"Z1", new List<object>()}
            };
            foreach (Curve curve in curves)
            {
                var p0 = curve.PointAtStart;
                var p1 = curve.PointAtEnd;

                dict["X0"].Add(p0.X); dict["Y0"].Add(p0.Y); dict["Z0"].Add(p0.Z);
                dict["X1"].Add(p1.X); dict["Y1"].Add(p1.Y); dict["Z1"].Add(p1.Z);
            }

            return dict;
        }

        private void CreateJsonFromDict(Dictionary<string, List<object>> dict, string dir, string name)
        { 
            //var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(dict);
            File.WriteAllText(Path.Combine(dir, name), json);

        }
        private Tuple<string, string> ExecutePython(string filePath, string pythonFile, string jsonName, string resultName)
        {
            // 1) Create Process Info
            var psi = new ProcessStartInfo();
            psi.FileName = @"C:\Users\sverremh\AppData\Local\Programs\Python\Python39\python.exe";

            // 2) Provide script and arguments
            var script = Path.Combine(filePath, "PythonFiles", pythonFile);
            var inputJson = Path.Combine(filePath, "Files", jsonName);
            var outputJson = Path.Combine(filePath, "Files", resultName);
            psi.Arguments = $"\"{script}\" \"{inputJson}\" \"{outputJson}\"";
            
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
            get { return new Guid("407936B7-8D89-42D8-A524-61929D9A4016"); }
        }
    }
}