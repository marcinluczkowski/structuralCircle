using System.Drawing;
using System.Reflection;

namespace StructuralCircleNTNU
{
    internal static class IconLoader
    {
        private static Bitmap _cachedIcon;

        public static Bitmap GetIcon()
        {
            if (_cachedIcon != null) return _cachedIcon;

            var assembly = Assembly.GetExecutingAssembly();
            using (var stream = assembly.GetManifestResourceStream("StructuralCircleNTNU.Resources.StructuralCircleIcon.png"))
            {
                if (stream != null)
                {
                    var original = new Bitmap(stream);
                    _cachedIcon = new Bitmap(original, 24, 24);
                }
            }
            return _cachedIcon;
        }
    }
}
