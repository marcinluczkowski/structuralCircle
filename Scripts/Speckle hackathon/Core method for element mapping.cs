using System;
using System.Collections;
using System.Collections.Generic;

using Rhino;
using Rhino.Geometry;

using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

using System.Linq;

/// <summary>
/// This class will be instantiated on demand by the Script component.
/// </summary>
public class Script_Instance : GH_ScriptInstance
{
#region Utility functions
  /// <summary>Print a String to the [Out] Parameter of the Script component.</summary>
  /// <param name="text">String to print.</param>
  private void Print(string text) { /* Implementation hidden. */ }
  /// <summary>Print a formatted String to the [Out] Parameter of the Script component.</summary>
  /// <param name="format">String format.</param>
  /// <param name="args">Formatting parameters.</param>
  private void Print(string format, params object[] args) { /* Implementation hidden. */ }
  /// <summary>Print useful information about an object instance to the [Out] Parameter of the Script component. </summary>
  /// <param name="obj">Object instance to parse.</param>
  private void Reflect(object obj) { /* Implementation hidden. */ }
  /// <summary>Print the signatures of all the overloads of a specific method to the [Out] Parameter of the Script component. </summary>
  /// <param name="obj">Object instance to parse.</param>
  private void Reflect(object obj, string method_name) { /* Implementation hidden. */ }
#endregion

#region Members
  /// <summary>Gets the current Rhino document.</summary>
  private readonly RhinoDoc RhinoDocument;
  /// <summary>Gets the Grasshopper document that owns this script.</summary>
  private readonly GH_Document GrasshopperDocument;
  /// <summary>Gets the Grasshopper script component that owns this script.</summary>
  private readonly IGH_Component Component;
  /// <summary>
  /// Gets the current iteration count. The first call to RunScript() is associated with Iteration==0.
  /// Any subsequent call within the same solution will increment the Iteration count.
  /// </summary>
  private readonly int Iteration;
#endregion

  /// <summary>
  /// This procedure contains the user code. Input parameters are provided as regular arguments,
  /// Output parameters as ref arguments. You don't have to assign output parameters,
  /// they will have a default value.
  /// </summary>
  private void RunScript(List<Line> bankLines, List<Line> objectLines, List<int> ib, List<int> jo, ref object wars, ref object ranking, ref object mappingLines, ref object newBank, ref object result)
  {
    int n = objectLines.Count;
    List<string> info = new List<string>();
    List<double> rank = new List<double>();
    List<int> map = new List<int>();
    List<Line> lines = new List<Line>();

    for (int i = 0; i < n; i++)
    {
      Line ol = objectLines[i];
      Line bl = bankLines[ib[i]];
      map.Add(ib[i]);
      double lol = ol.Length;
      double lbl = bl.Length;

      double dif = lbl - lol;
      info.Add("dif = " + dif);
      if(dif < 0)
      {
        //to small element, we have to punish it
        rank.Add(50);
      }
      else if (dif == 0)
      {
        //perfect match
        rank.Add(0);
      }
      else if (dif < 900 && dif > 0)
      {
        //not so bad match the waste is less than 0.9m
        rank.Add(5);
      }
      else if (dif >= 900 && dif > 0)
      {
        //not so bad match the waste is more than 0.9m
        rank.Add(2);
        bankLines.Add(new Line(new Point3d(0, 0, 0), new Point3d(dif, 0, 0)));
      }

      lines.Add(new Line(ol.PointAt(0.5), bl.PointAt(0.5)));
    }


    int nm = map.Count;
    List<int> mapd = map.Distinct().ToList();
    int nmd = mapd.Count;
    info.Add("nm = " + nm);
    info.Add("nmd = " + nmd);
    //extra punishment for using the same elements from bank to object
    rank.Add(Convert.ToDouble(nm - nmd) * 100);


    foreach(var im in map)
      bankLines.RemoveAt(im);

    double res = 0;
    foreach(var r in rank)
      res = res + r;



    //output
    result = res;
    ranking = rank;
    wars = info;
    newBank = bankLines;
    mappingLines = lines;
  }

  // <Custom additional code> 

  // </Custom additional code> 
}