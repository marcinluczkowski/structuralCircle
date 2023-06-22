using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Eventing.Reader;
using System.Linq;

using Grasshopper;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Special;
using Grasshopper.Kernel.Types;
using System.IO;
using System.Text.Json;
using System.Text.RegularExpressions;
using Serilog;
using System.Reflection;
using Rhino.Input.Custom;
using System.Text;
using System.Threading;

namespace MatchingWrapper
{
    public static class HelperMethods
    {
        /// <summary>
        /// Converts a Grasshopper DataTree to a nested list in C#
        /// </summary>
        /// <param name="tree">DataTree to convert.</param>
        /// <returns>Nested list.</returns>
        public static List<List<object>> TreeToNestedList(GH_Structure<IGH_Goo> tree)
        {
            var nestedList = new List<List<object>>();

            foreach (var branch in tree.Branches)
            {
                var objList = new List<object>();
                object obj;
                foreach (var goo in branch)
                {
                    goo.CastTo(out obj);
                    objList.Add(obj);

                }
                nestedList.Add(objList);
            }

            return nestedList;
        }
        /// <summary>
        /// Converts a Nested List to a DataTree structure for Grasshopper
        /// </summary>
        /// <param name="nestedList">Nested list to convert</param>
        /// <returns>A grasshopper DataTree<object></returns>
        public static DataTree<object> NestedListToDataTree(List<List<object>> nestedList)
        {
            var dataTree = new DataTree<object>();
            int num = 0;
            foreach (var lst in nestedList)
            {
                
                dataTree.AddRange(lst, new GH_Path(num));
                num++;
            }
            return dataTree;
        }

        public static List<List<string>> CsvToNestedList(string filepath)
        {
            List<List<string>> nestedList = new List<List<string>>();
            using (var reader = new StreamReader(filepath))
            {
                
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var list = line.Split(',').ToList();
                    nestedList.Add(list);
                }

                
            }
            nestedList.RemoveAt(0);
            return nestedList;
        }

        public static void JsonFromList(List<List<object>> nestedList, string path, string filename)
        {
            var json = JsonSerializer.Serialize(nestedList);
            File.WriteAllText(Path.Combine(path, filename), json);
        }

        public static void CreateJsonFromDict(Dictionary<string, List<object>> dict, string dir, string name)
        {
            //var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(dict);
            File.WriteAllText(Path.Combine(dir, name), json);

        }
        public static Tuple<string, string> ExecutePython(string filePath, string pythonFile, string demandName, string supplyName, string resultName)
        {
            // 1) Create Process Info
            var psi = new ProcessStartInfo();
            //psi.FileName = @"C:\Users\sverremh\AppData\Local\Programs\Python\Python39\python.exe"; // OBS: This will only work for me. Should be more general
            var username = Environment.UserName;
            psi.FileName = String.Format(@"C:\Users\{0}\AppData\Local\Programs\Python\Python39\python.exe", username);
            // 2) Provide script and arguments
            var script = Path.Combine(filePath, "PythonFiles", pythonFile);
            var demandJson = Path.Combine(filePath, "Files", demandName);
            var supplyJson = Path.Combine(filePath, "Files", supplyName);
            var outputJson = Path.Combine(filePath, "Files", resultName);
            var args = new List<string>() { script, demandJson, supplyJson, outputJson };
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


        /// <summary>
        /// New version of Execute method that allows for a python file from a completely different path
        /// </summary>
        /// <param name="filePath"></param>
        /// <param name="pythonFile"></param>
        /// <param name="demandName"></param>
        /// <param name="supplyName"></param>
        /// <param name="resultName"></param>
        /// <returns></returns>
        public static Tuple<string, string> ExecutePython2(string filePath, string fullPythonPath, string methodName, string demandName, string supplyName, string resultName, string constraints)
        {
            // 1) Create Process Info
            ProcessStartInfo psi = new ProcessStartInfo();
            
            // create the arguments
            var username = Environment.UserName;
            psi.FileName = String.Format(@"C:\Users\{0}\AppData\Local\Programs\Python\Python39\python.exe", username); // Location of python.exe
            // 2) Provide script and arguments
            var script = fullPythonPath;
            var demandJson = Path.Combine(filePath, "Files", demandName);
            var supplyJson = Path.Combine(filePath, "Files", supplyName);
            var outputJson = Path.Combine(filePath, "Files", resultName);
            var args = new List<string>() { script, methodName, demandJson, supplyJson, outputJson  ,constraints};
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

            using (Process process = Process.Start(psi))
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


        public static string ExecuteBatch(string filePath = "", string PythonPath = "", int methodNumber = 0, string demandPath = "", string supplyPath = "", string constraints = "")
        {
            try
            {
                int exitCode;
                var username = Environment.UserName; // Get the username of the current user.
                string batchFilePath = String.Format(@"C:\Users\{0}\AppData\Roaming\Grasshopper\Libraries\MatchingWrapper\PythonFiles\test.bat", username); // location to runfile

                

                Process process = new Process();
                process.StartInfo.FileName = "cmd.exe";
                process.StartInfo.Arguments = "/c " + batchFilePath + " \"" +  Convert.ToString(methodNumber) + "\" \"" + demandPath + "\" \"" + supplyPath + "\" \"" + constraints + "\"";
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.RedirectStandardError = true;
                

                process.Start();
                string output = process.StandardOutput.ReadToEnd();
                string errors = process.StandardError.ReadToEnd();
                process.WaitForExit();
                
                //Console.WriteLine("Batch file execution completed.");
                //Console.WriteLine("Output:");
                //Console.WriteLine(output);

                return output;
            }
            catch (Exception ex)
            {
                return "Error occurred: " + ex.Message;
            }

        }

        public static string ExecuteBatch2(string filePath = "", string PythonPath = "", int methodNumber = 0, string demandPath = "", string supplyPath = "")
        {
            // 1) create the process start info
            int exitCode;
            var username = Environment.UserName; // Get the username of the current user.
            var batchFileLocation = String.Format(@"C:\Users\{0}\AppData\Roaming\Grasshopper\Libraries\MatchingWrapper\PythonFiles\test.bat", username); // location to runfile



            // 2) Provide script and arguments
            // create command line command
            //string argString = $"/c";
            string argString = "";
            //var args = new List<string> { batchFileLocation, String.Format("{0}", methodNumber), "3" };
            var args = new List<string> { batchFileLocation };
            
            foreach (string arg in args)
            {
                argString += (" " + arg + " ");
            }

            // 3) Process configuration
            string filename = "cmd.exe";

            // 4) Execute process and get output

            // 4) Execute process and get output
            //List<string> errors = new List<string>();
            //List<string> results = new List<string>();
            string outputString = "";
            using (Process process = new Process())
            {
                process.StartInfo.FileName = filename;
                process.StartInfo.Arguments = argString;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.CreateNoWindow = true;
                process.StartInfo.RedirectStandardError = true;
                process.StartInfo.RedirectStandardOutput = true;

                StringBuilder output = new StringBuilder();
                StringBuilder error = new StringBuilder();

                using (AutoResetEvent outputWaitHandle = new AutoResetEvent(false))
                using (AutoResetEvent errorWaitHandle = new AutoResetEvent(false))
                {
                    process.OutputDataReceived += (sender, e) =>
                    {
                        if (e.Data == null)
                        {
                            outputWaitHandle.Set();
                        }
                        else
                        {
                            output.AppendLine(e.Data);
                        }
                        outputString = output.ToString();
                    };
                    process.ErrorDataReceived += (sender, e) =>
                    {
                        if (e.Data == null)
                        {
                            errorWaitHandle.Set();
                        }
                        else
                        {
                            error.AppendLine(e.Data);
                        }
                    };

                    process.Start();
                    double maxWaitMin = 0.25; // number of minutes to wait for solver to run. 
                    int timeout = Convert.ToInt32(1000 * 60 * maxWaitMin);
                    bool b1 = process.WaitForExit(timeout);
                    bool b2 = outputWaitHandle.WaitOne(timeout);
                    bool b3 = errorWaitHandle.WaitOne(timeout);
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();
                    
                    if (b1 && b2 && b3)
                    {
                        // Process completed. Check process.ExitCode here.
                        string test = process.StandardOutput.ReadToEnd();
                        exitCode = process.ExitCode;
                    }
                    else
                    {
                        string timeOutString = "Something happened?";
                        // Timed out.
                    }
                }
                /*
                try
                {
                    var w = new System.Diagnostics.Stopwatch();
                    Log.Information("Reading results from shell:"); w.Start();
                    
                    while (process.StandardError.Peek() > -1)
                    {
                        error.Add(process.StandardError.ReadLine());
                    }
                    
                    while (process.StandardOutput.Peek() > -1)
                    {
                        output.Add(process.StandardOutput.ReadLine());
                    }

                    
                    exitCode = process.ExitCode;
                    Log.Debug("Time used to read results from console: {time} ms", w.ElapsedMilliseconds); w.Reset();
                }
                catch (Exception e)
                {
                    exitCode = 1;
                }
            }
            if (exitCode == 0)
            {
                return results;
            }
            else
            {
                return results;
            }
                */

            }

            return outputString;
        }


        public static List<int?> DeserializeJson(string path)
        {
            var json = File.ReadAllText(path);
            var vals = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json);
            

            var els = vals["Supply_id"].ToString();
            var lst = els.Split(',').ToList();
            lst = lst.Select(x => x.Split(':')[1]).ToList();
            lst[lst.Count - 1] = Regex.Replace(lst[lst.Count - 1], "[}]", string.Empty);

            // cast all elements to integers:
            List<int?> mappingId = lst.Select(x => Int32.TryParse(x, out var tempVal) ? tempVal : (int?)null).ToList();

            /*
            foreach (var kvp in vals)
            {
                var val = vals[kvp.Key].ToString();
                var strList = val.Split(',').Select(x => x.Split(':')[1]).ToList();
                strList[strList.Count - 1] = Regex.Replace(strList[strList.Count - 1], "[}]", string.Empty);                
            }
            */

            return mappingId;
        }
    }
}
