using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Grasshopper.Kernel;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    /// <summary>
    /// Parses CSV files into Element lists.
    ///
    /// Format A - Material List (auto-detect beam vs plate):
    ///   MaterialType,Width,Height,Length,Quantity
    ///   All geometry columns (Width, Height, Length) use the same CSV length unit (mm, cm, or m) and are converted to metres.
    ///   Element type is inferred from the material name:
    ///     - "Limtre", "GL", "KVH", "C24" narrow section → Beam
    ///     - "X-LAM", "CLT", "Mass timber", wide Width (≥ 300mm) → Plate
    ///
    /// Format B - Manual element list (explicit type column):
    ///   Id,Name,Location,Material,Width,Height,Length,Iy,Iz   → Beam
    ///   Id,Name,Location,Material,Thickness                   → Plate
    /// </summary>
    public static class CsvParser
    {
        /// <summary>
        /// Converts a CSV length unit string to a multiplier (value × scale → metres).
        /// Accepts mm, cm, m (case-insensitive). Unknown values default to m (scale 1) with optional warning.
        /// </summary>
        public static double LengthScaleToMetres(string unit, GH_Component component = null)
        {
            if (string.IsNullOrWhiteSpace(unit))
                return 1.0;

            switch (unit.Trim().ToLowerInvariant())
            {
                case "mm": return 0.001;
                case "cm": return 0.01;
                case "m": return 1.0;
                default:
                    component?.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                        $"Unknown length unit '{unit}'. Use mm, cm, or m. Assuming metres.");
                    return 1.0;
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Format A: MaterialType,Width,Height,Length,Quantity
        // All geometry columns use the same <paramref name="lengthUnit"/> (scaled to metres).
        // ─────────────────────────────────────────────────────────────────

        public static List<Element> ParseMaterialList(string filePath, GH_Component component, string lengthUnit = "mm")
        {
            var elements = new List<Element>();
            string[] lines = File.ReadAllLines(filePath);

            if (lines.Length < 2)
            {
                component?.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "CSV file has no data rows");
                return elements;
            }

            string[] headers = lines[0].Split(',');
            var headerMap = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < headers.Length; i++)
                headerMap[headers[i].Trim()] = i;

            bool hasMaterialType = headerMap.ContainsKey("MaterialType");
            bool hasWidth        = headerMap.ContainsKey("Width");
            bool hasHeight       = headerMap.ContainsKey("Height");
            bool hasLength       = headerMap.ContainsKey("Length");
            bool hasQuantity     = headerMap.ContainsKey("Quantity");

            if (!hasMaterialType || !hasWidth || !hasHeight || !hasLength)
            {
                component?.AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                    "Expected columns: MaterialType,Width,Height,Length,Quantity");
                return null;
            }

            double scale = LengthScaleToMetres(lengthUnit, component);
            int globalId = 0;

            for (int row = 1; row < lines.Length; row++)
            {
                string line = lines[row].Trim();
                if (string.IsNullOrEmpty(line)) continue;

                string[] cols = line.Split(',');

                try
                {
                    string matType   = GetString(cols, headerMap, "MaterialType", "Unknown");
                    double widthRaw  = GetDouble(cols, headerMap, "Width");
                    double heightRaw = GetDouble(cols, headerMap, "Height");
                    double lengthRaw = GetDouble(cols, headerMap, "Length");
                    int    quantity  = hasQuantity ? GetInt(cols, headerMap, "Quantity", 1) : 1;

                    double widthM  = widthRaw * scale;
                    double heightM = heightRaw * scale;
                    double lengthM = lengthRaw * scale;

                    bool isPlate = DetectPlate(matType, widthM, heightM);

                    var material = new Material(globalId, matType);

                    for (int q = 0; q < quantity; q++, globalId++)
                    {
                        string elemName = $"{matType}_{widthRaw:0}x{heightRaw:0}_{lengthM:F3}_{globalId}";

                        if (isPlate)
                        {
                            // For CLT/X-LAM: Width = panel width, Height = thickness
                            var section = new PlateSection(globalId, $"T{heightRaw:0}", heightM);
                            var plate = new Plate(globalId, elemName, "", material, section, null);
                            plate.AxisLine = new Line(Point3d.Origin, new Point3d(lengthM, 0, 0));
                            elements.Add(plate);
                        }
                        else
                        {
                            // For Glulam/Beam: Width x Height cross-section
                            var section = new BeamSection(globalId, $"{widthRaw:0}x{heightRaw:0}", widthM, heightM);
                            var axis    = new Line(Point3d.Origin, new Point3d(lengthM, 0, 0));
                            var beam    = new Beam(globalId, elemName, "", material, section, axis);
                            elements.Add(beam);
                        }
                    }
                }
                catch (Exception ex)
                {
                    component?.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Row {row}: {ex.Message}");
                }
            }

            return elements;
        }

        /// <summary>
        /// Detect whether an entry is a plate/slab.
        /// Rules (in priority order):
        ///  1. Material name contains "X-LAM" or "CLT" or "XLAM" → Plate
        ///  2. Width ≥ 500 mm AND Width > Height → Plate (wide flat panel)
        ///  3. Otherwise → Beam
        /// </summary>
        /// <param name="widthM">Section width in metres.</param>
        /// <param name="heightM">Section height/thickness in metres.</param>
        static bool DetectPlate(string matType, double widthM, double heightM)
        {
            string upper = matType.ToUpperInvariant();
            if (upper.Contains("X-LAM") || upper.Contains("XLAM") || upper.Contains("CLT") || upper.Contains("KERTO-Q"))
                return true;

            if (widthM >= 0.5 && widthM > heightM)
                return true;

            return false;
        }

        // ─────────────────────────────────────────────────────────────────
        // Format B: explicit type passed by component
        // ─────────────────────────────────────────────────────────────────

        public static List<Element> ParseElements(string filePath, string type, GH_Component component, string lengthUnit = "m")
        {
            var elements = new List<Element>();
            string[] lines = File.ReadAllLines(filePath);
            double scale = LengthScaleToMetres(lengthUnit, component);

            if (lines.Length < 2)
            {
                component?.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "CSV file has no data rows");
                return elements;
            }

            string[] headers = lines[0].Split(',');
            var headerMap = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < headers.Length; i++)
                headerMap[headers[i].Trim()] = i;

            for (int row = 1; row < lines.Length; row++)
            {
                string line = lines[row].Trim();
                if (string.IsNullOrEmpty(line)) continue;

                string[] cols = line.Split(',');

                try
                {
                    if (type.Equals("Beam", StringComparison.OrdinalIgnoreCase))
                    {
                        var elem = ParseBeamRow(cols, headerMap, row, scale);
                        if (elem != null) elements.Add(elem);
                    }
                    else if (type.Equals("Plate", StringComparison.OrdinalIgnoreCase))
                    {
                        var elem = ParsePlateRow(cols, headerMap, row, scale);
                        if (elem != null) elements.Add(elem);
                    }
                    else
                    {
                        component?.AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                            $"Unknown type: {type}. Use 'Beam' or 'Plate'.");
                        return null;
                    }
                }
                catch (Exception ex)
                {
                    component?.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Row {row}: {ex.Message}");
                }
            }

            return elements;
        }

        static Beam ParseBeamRow(string[] cols, Dictionary<string, int> headers, int rowIndex, double lengthScaleToM)
        {
            int    id       = GetInt(cols, headers, "Id", rowIndex);
            string name     = GetString(cols, headers, "Name", $"Beam_{rowIndex}");
            string location = GetString(cols, headers, "Location", "");
            string matName  = GetString(cols, headers, "Material", "Timber");
            double width    = GetDouble(cols, headers, "Width") * lengthScaleToM;
            double height   = GetDouble(cols, headers, "Height") * lengthScaleToM;
            double length   = GetDouble(cols, headers, "Length") * lengthScaleToM;

            var material = new Material(id, matName);
            BeamSection section;

            if (headers.ContainsKey("Iy") && headers.ContainsKey("Iz"))
                section = new BeamSection(id, $"{width}x{height}", width, height,
                              GetDouble(cols, headers, "Iy"), GetDouble(cols, headers, "Iz"));
            else
                section = new BeamSection(id, $"{width}x{height}", width, height);

            return new Beam(id, name, location, material, section,
                            new Line(Point3d.Origin, new Point3d(length, 0, 0)));
        }

        static Plate ParsePlateRow(string[] cols, Dictionary<string, int> headers, int rowIndex, double lengthScaleToM)
        {
            int    id        = GetInt(cols, headers, "Id", rowIndex);
            string name      = GetString(cols, headers, "Name", $"Plate_{rowIndex}");
            string location  = GetString(cols, headers, "Location", "");
            string matName   = GetString(cols, headers, "Material", "Timber");
            double thickness = GetDouble(cols, headers, "Thickness") * lengthScaleToM;

            var material = new Material(id, matName);
            PlateSection section;
            if (headers.ContainsKey("Width"))
            {
                double w = GetDouble(cols, headers, "Width") * lengthScaleToM;
                section = new PlateSection(id, $"T={thickness}", thickness, w);
            }
            else
                section = new PlateSection(id, $"T={thickness}", thickness);

            double axisLen = headers.ContainsKey("Length")
                ? GetDouble(cols, headers, "Length") * lengthScaleToM
                : 0;
            var plate = new Plate(id, name, location, material, section, null);
            if (axisLen > 1e-9)
                plate.AxisLine = new Line(Point3d.Origin, new Point3d(axisLen, 0, 0));
            return plate;
        }

        // ─────────────────────────────────────────────────────────────────
        // Helpers
        // ─────────────────────────────────────────────────────────────────

        static int GetInt(string[] cols, Dictionary<string, int> headers, string key, int fallback)
        {
            if (headers.TryGetValue(key, out int idx) && idx < cols.Length)
                if (int.TryParse(cols[idx].Trim(), out int v)) return v;
            return fallback;
        }

        static string GetString(string[] cols, Dictionary<string, int> headers, string key, string fallback)
        {
            if (headers.TryGetValue(key, out int idx) && idx < cols.Length)
            {
                string v = cols[idx].Trim();
                return string.IsNullOrEmpty(v) ? fallback : v;
            }
            return fallback;
        }

        static double GetDouble(string[] cols, Dictionary<string, int> headers, string key)
        {
            if (headers.TryGetValue(key, out int idx) && idx < cols.Length)
                if (double.TryParse(cols[idx].Trim(), NumberStyles.Any,
                    CultureInfo.InvariantCulture, out double v)) return v;
            return 0.0;
        }
    }
}
