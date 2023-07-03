using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Grasshopper;
using Rhino.Geometry;
using System.Security.Principal;
using System.Text;
using System.IO;
using MatchingWrapper.Properties;

namespace MatchingWrapper
{
    public class AddMaterialLocation : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the AddMaterialLocation class.
        /// </summary>
        public AddMaterialLocation()
          : base("AddMaterialLocation", "addMat",
              "Description",
              "Python", "Misc")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("Location Name", "location", "Name of the location", GH_ParamAccess.item); // 0
            pManager.AddTextParameter("Material Name", "material", "Name of the material", GH_ParamAccess.item); // 1
            pManager.AddNumberParameter("Latitude", "lat", "Latitude of the location", GH_ParamAccess.item); // 2
            pManager.AddNumberParameter("Longitude", "long", "Longitude of the locatio", GH_ParamAccess.item); // 3
            pManager.AddBooleanParameter("Save", "save", "Writes the material location to the database", GH_ParamAccess.item); // 4
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {                    

        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object is used to retrieve from inputs and store in outputs.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            // -- input --
            string location = ""; // 0
            string material = ""; // 1
            double latitude = 0.0; // 2
            double longitude = 0.0; // 3
            bool run = false; // 4

            // File path of the demand path. Should probably move it to the Grasshopper Library package. Not within the python package
            string filePath = String.Format("C:\\Users\\{0}\\AppData\\Roaming\\Grasshopper\\Libraries\\MatchingWrapper\\PythonFiles\\matching_env\\Lib\\site-packages\\elementmatcher\\data\\CSV",
                Environment.UserName);
            if (!DA.GetData(4, ref run)) return;
            // If run, write to file
            if (run)
            {
                // Assign all material
                if (!DA.GetData(0, ref location))
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "You need to specify the Location Name");
                    return;
                }
                if (!DA.GetData(1, ref material))
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "You need to specify the Material Name");
                    return;
                }
                if (!DA.GetData(2, ref latitude))
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "You need to specify the Latitude");
                    return;
                }
                if (!DA.GetData(3, ref longitude))
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "You need to specify the Longitude");
                    return;
                }
                // Don't have write access to the csv when inside the python packages. Move outside.
                File.AppendAllText(filePath, String.Format("{0},{1},{2},{3}", location, material, latitude, longitude));

                AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "Material location sucessfully added to database");


            }
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
                return Resources.add_location;
            }
        }

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("82FE7AB1-18FB-4740-B429-B6C64EBD3321"); }
        }
    }
}