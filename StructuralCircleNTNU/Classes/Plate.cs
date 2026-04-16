using System;
using Rhino.Geometry;

namespace StructuralCircleNTNU.Classes
{
    public class Plate : Element
    {
        public override string ElementType => "Plate";

        public new PlateSection Section
        {
            get => base.Section as PlateSection;
            set => base.Section = value;
        }

        /// <summary>Length along the axis line (if set), otherwise 0.</summary>
        public double Length => (AxisLine.IsValid && AxisLine.Length > 0) ? AxisLine.Length : 0;

        public Plate() { }

        public Plate(int id, string name, string location, Material material, PlateSection section, Surface axisSurface)
            : base(id, name, location, material, section)
        {
            AxisSurface = axisSurface;
        }

        public override string ToString()
        {
            string sec = Section != null ? $", {Section}" : "";
            string mat = Material != null ? $", {Material}" : "";
            return $"Plate [{Id}]: {Name} (T={Section?.Thickness:F3}, L={Length:F3}{sec}{mat})";
        }
    }
}
