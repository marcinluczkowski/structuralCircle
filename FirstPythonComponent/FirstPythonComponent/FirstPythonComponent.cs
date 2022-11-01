using Grasshopper;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;

namespace FirstPythonComponent
{
    public class FirstPythonComponent : GH_Component
    {
        /// <summary>
        /// Each implementation of GH_Component must provide a public 
        /// constructor without any arguments.
        /// Category represents the Tab in which the component will appear, 
        /// Subcategory the panel. If you use non-existing tab or panel names, 
        /// new tabs/panels will automatically be created.
        /// </summary>
        public FirstPythonComponent()
          : base("FirstPythonComponent", "ASpi",
            "Construct an Archimedean, or arithmetic, spiral given its radii and number of turns.",
            "Python", "Primitive")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            // Use the pManager object to register your input parameters.
            // You can often supply default values when creating parameters.
            // All parameters must have the correct access type. If you want 
            // to import lists or trees of values, modify the ParamAccess flag.
            pManager.AddNumberParameter("Number 1", "num1", "First number", GH_ParamAccess.item, 1);
            pManager.AddNumberParameter("Number 2", "num2", "Second number", GH_ParamAccess.item, 1);

            // If you want to change properties of certain parameters, 
            // you can use the pManager instance to access them by index:
            //pManager[0].Optional = true;
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            // Use the pManager object to register your output parameters.
            // Output parameters do not have default values, but they too must have the correct access type.
            pManager.AddTextParameter("Result", "rslt", "The result of the operation", GH_ParamAccess.item); 

            // Sometimes you want to hide a specific parameter from the Rhino preview.
            // You can use the HideParameter() method as a quick way:
            //pManager.HideParameter(0);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="da">The DA object can be used to retrieve data from input parameters and 
        /// to store data in output parameters.</param>
        protected override void SolveInstance(IGH_DataAccess da)
        {
            // First, we need to retrieve all data from the input parameters.
            // We'll start by declaring variables and assigning them starting values.
            double num1 = 0.0; 
            double num2 = 0.0;


            // Then we need to access the input parameters individually. 
            // When data cannot be extracted from a parameter, we should abort this method.
            if (!da.GetData(0, ref num1)) return;
            if (!da.GetData(1, ref num2)) return;

            

            // Here we solve the method
            var result = ExcecutePython(num1, num2);

            // Finally assign the spiral to the output parameter.
            da.SetData(0, result.Item2);
        }

        #region Methods

        public Tuple<string, string> ExcecutePython(double v1, double v2)
        {
            // 1) Create Process Info
            var psi = new ProcessStartInfo();
            psi.FileName = @"C:\Users\sverremh\AppData\Local\Programs\Python\Python39\python.exe";

            // 2) Provide script and arguments
            List<double> numbers = new List<double>(){1, 2, 3, 4, 5};
            var script = @"C:\Users\sverremh\source\repos\FirstPythonComponent\FirstPythonComponent\firstPython.py";
            psi.Arguments = $"\"{script}\" \"{numbers}\" \"{v2}\"";

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

                    errors = process.StandardError.ReadToEnd();
                    
                    process.BeginOutputReadLine();
                    results = process.StandardOutput.ReadToEnd();
                }
                catch (Exception e)
                {

                }

            }
            return new Tuple<string, string>(errors, results);
        }

        #endregion


        /// <summary>
        /// The Exposure property controls where in the panel a component icon 
        /// will appear. There are seven possible locations (primary to septenary), 
        /// each of which can be combined with the GH_Exposure.obscure flag, which 
        /// ensures the component will only be visible on panel dropdowns.
        /// </summary>
        public override GH_Exposure Exposure => GH_Exposure.primary;

        /// <summary>
        /// Provides an Icon for every component that will be visible in the User Interface.
        /// Icons need to be 24x24 pixels.
        /// You can add image files to your project resources and access them like this:
        /// return Resources.IconForThisComponent;
        /// </summary>
        protected override System.Drawing.Bitmap Icon => null;

        /// <summary>
        /// Each component must have a unique Guid to identify it. 
        /// It is vital this Guid doesn't change otherwise old ghx files 
        /// that use the old ID will partially fail during loading.
        /// </summary>
        public override Guid ComponentGuid => new Guid("61FEAF66-6575-40BC-A092-3F370603A94D");
    }
}