using System;

namespace Oxel
{
    //           ^
    //           |       polygon
    // next      |
    // half-edge |
    //           |       half-edge
    // vertex	 *<====================== 
    //           ---------------------->*
    //				  twin-half-edge
    public class HalfEdge
    {
        public short NextIndex;
        public short TwinIndex;
        public short VertexIndex;
        public short PolygonIndex;
    }
}