using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Windows.Forms;
using System.Xml;

namespace Oxel
{
    public class UpdateInformation
    {
        public Version Version = null;
        public string URL = "";
    }

    public static class UpdateChecker
    {
        public static void PerformCheck()
        {
            UpdateInformation info = Check();
            if (info == null)
                return;

            Version curVersion = Assembly.GetExecutingAssembly().GetName().Version;
            if (curVersion.CompareTo(info.Version) < 0)
            {
                string title = "New version detected.";
                string question = "Download the new version?";
                if (DialogResult.Yes == 
                    MessageBox.Show(question, title, MessageBoxButtons.YesNo, MessageBoxIcon.Question))
                {
                    System.Diagnostics.Process.Start(info.URL);
                }
            }
        }

        private static UpdateInformation Check()
        {
            UpdateInformation info = null;

            XmlTextReader reader = null;
            try
            {
                string xmlURL = "http://bitbucket.org/NickDarnell/oxel/downloads/oxel_update.xml";
                reader = new XmlTextReader(xmlURL);
                reader.MoveToContent();
                string elementName = "";

                if (reader.NodeType == XmlNodeType.Element && reader.Name == "oxel")
                {
                    while (reader.Read())
                    {
                        if (reader.NodeType == XmlNodeType.Element)
                        {
                            elementName = reader.Name;
                        }
                        else
                        {
                            if (reader.NodeType == XmlNodeType.Text && reader.HasValue)
                            {
                                switch (elementName)
                                {
                                    case "version":
                                        info.Version = new Version(reader.Value);
                                        break;
                                    case "url":
                                        info.URL = reader.Value;
                                        break;
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception)
            {
            }
            finally
            {
                if (reader != null)
                    reader.Close();
            }

            return info;
        }
    }
}