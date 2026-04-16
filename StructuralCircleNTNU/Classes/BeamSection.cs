using System;

namespace StructuralCircleNTNU.Classes
{
    public class BeamSection : Section
    {
        public double Area { get; set; }
        public double Iy { get; set; }
        public double Iz { get; set; }
        public double Width { get; set; }
        public double Height { get; set; }

        public BeamSection() { }

        public BeamSection(int id, string name, double width, double height, double iy, double iz)
            : base(id, name)
        {
            Width = width;
            Height = height;
            Area = width * height;
            Iy = iy;
            Iz = iz;
        }

        public BeamSection(int id, string name, double width, double height)
            : base(id, name)
        {
            Width = width;
            Height = height;
            Area = width * height;
            Iy = (width * Math.Pow(height, 3)) / 12.0;
            Iz = (height * Math.Pow(width, 3)) / 12.0;
        }

        public override string ToString()
        {
            return $"BeamSection [{Id}]: {Name} (W={Width:F3}, H={Height:F3}, A={Area:F4}, Iy={Iy:E2}, Iz={Iz:E2})";
        }
    }
}
