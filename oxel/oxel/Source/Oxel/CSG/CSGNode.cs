using System;
using System.Linq;
using System.Diagnostics;
using System.Collections.Generic;
using OpenTK;

namespace Oxel
{
    [DebuggerDisplay("{NodeType}({Left},{Right})")]
    public sealed class CSGNode
    {
        public readonly AABBi Bounds = new AABBi();
        public readonly CSGNodeType NodeType;

        public CSGNode Left;
        public CSGNode Right;
        public CSGNode Parent;

        public Vector3 LocalTranslation;
        public Vector3 Translation;
        public Plane[] Planes;

        public CSGNode(CSGNodeType branchOperator)
        {
            NodeType = branchOperator;
            Left = null;
            Right = null;
        }

        public CSGNode(CSGNodeType branchOperator, CSGNode left, CSGNode right)
        {
            NodeType = branchOperator;
            Left = left;
            Right = right;
        }

        public CSGNode(IEnumerable<Plane> planes)
        {
            this.NodeType = CSGNodeType.Brush;
            Planes = planes.ToArray();
        }
    }
}
