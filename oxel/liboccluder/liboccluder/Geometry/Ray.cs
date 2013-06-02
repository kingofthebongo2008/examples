using OpenTK;

namespace Oxel
{
    public struct Ray
    {
        public Vector3 Origin;
        public Vector3 Normal;

        public Ray(Vector3 origin, Vector3 normal)
        {
            this.Origin = origin;
            this.Normal = normal;
        }

        public Ray(ref Vector3 origin, ref Vector3 normal)
        {
            this.Origin = origin;
            this.Normal = normal;
        }

        public bool Intersection(Plane plane, out Vector3 intersection)
        {
            float normDot = Vector3.Dot(Normal, plane.Normal);
            if (normDot == 0)
            {
                intersection = Vector3.Zero;
                return false;
            }

            float t = -(Vector3.Dot(Origin, plane.Normal) + plane.D) / normDot;
            if (t <= 0)
            {
                intersection = Vector3.Zero;
                return false;
            }

            intersection = Origin + t * Normal;
                return true;
        }
    }
}
