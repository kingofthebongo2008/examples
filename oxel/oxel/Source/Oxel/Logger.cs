using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace Oxel
{
    public static class Logger
    {
        public static bool IsCommandLine;

        public static void DisplayError(string message)
        {
            if (IsCommandLine)
                Console.WriteLine("Oxel Error: " + message);
            else
                MessageBox.Show(message, "Oxel Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}
