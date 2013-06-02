using System;
using System.IO;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.ComponentModel;

namespace Oxel
{
    /// <summary>
    /// A command line parer class.
    /// </summary>
    /// <remarks>
    //  The original version of this class came from Shawn Hargreaves blog,
    /// http://blogs.msdn.com/b/shawnhar/archive/2012/04/20/a-reusable-reflection-based-command-line-parser.aspx
    /// </remarks>
    public class CommandLineParser
    {
        object optionsObject;

        Queue<FieldInfo> requiredOptions = new Queue<FieldInfo>();
        Dictionary<string, FieldInfo> optionalOptions = new Dictionary<string, FieldInfo>();

        List<string> requiredUsageHelp = new List<string>();
        List<string> optionalUsageHelp = new List<string>();

        public static bool Parse<T>(string[] args, ref T options) where T : class, new()
        {
            CommandLineParser parser = new CommandLineParser(options);

            // Parse each argument in turn.
            foreach (string arg in args)
            {
                if (arg == "-h" || arg == "-help")
                {
                    parser.ShowError("Displaying Help");
                    return false;
                }

                if (!parser.ParseArgument(arg.Trim()))
                {
                    return false;
                }
            }

            // Make sure we got all the required options.
            FieldInfo missingRequiredOption = parser.requiredOptions.FirstOrDefault(field => !IsList(field) || parser.GetList(field).Count == 0);

            if (missingRequiredOption != null)
            {
                parser.ShowError("Missing argument '{0}'", GetOptionName(missingRequiredOption));
                return false;
            }

            return true;
        }

        public static T Parse<T>(string[] args) where T : class, new()
        {
            T optionsObject = new T();
            CommandLineParser parser = new CommandLineParser(optionsObject);

            // Parse each argument in turn.
            foreach (string arg in args)
            {
                if (arg == "-h" || arg == "-help")
                {
                    parser.ShowError("Displaying Help");
                    return null;
                }

                if (!parser.ParseArgument(arg.Trim()))
                {
                    return null;
                }
            }

            // Make sure we got all the required options.
            FieldInfo missingRequiredOption = parser.requiredOptions.FirstOrDefault(field => !IsList(field) || parser.GetList(field).Count == 0);

            if (missingRequiredOption != null)
            {
                parser.ShowError("Missing argument '{0}'", GetOptionName(missingRequiredOption));
                return null;
            }

            return optionsObject;
        }

        protected CommandLineParser(object optionsObject)
        {
            this.optionsObject = optionsObject;

            // Reflect to find what command line options are available.
            foreach (FieldInfo field in optionsObject.GetType().GetFields())
            {
                string fieldName = GetOptionName(field);
                string description = GetOptionDescription(field);

                if (GetAttribute<IgnoreAttribute>(field) != null) continue;

                if (GetAttribute<RequiredAttribute>(field) != null)
                {
                    // Record a required option.
                    requiredOptions.Enqueue(field);

                    requiredUsageHelp.Add(string.Format("<{0}> | {1}", fieldName, description));
                }
                else
                {
                    // Record an optional option.
                    optionalOptions.Add(fieldName.ToLowerInvariant(), field);

                    if (field.FieldType == typeof(bool))
                    {
                        optionalUsageHelp.Add(string.Format("-{0} | {1}", fieldName, description));
                    }
                    else
                    {
                        optionalUsageHelp.Add(string.Format("-{0}=value | {1}", fieldName, description));
                    }
                }
            }
        }

        bool ParseArgument(string arg)
        {
            if (arg.StartsWith("-"))
            {
                // Parse an optional argument.
                char[] separators = { '=' };

                string[] split = arg.Substring(1).Split(separators, 2, StringSplitOptions.None);

                string name = split[0];
                string value = (split.Length > 1) ? split[1] : "true";

                FieldInfo field;

                if (!optionalOptions.TryGetValue(name.ToLowerInvariant(), out field))
                {
                    ShowError("Unknown option '{0}'", name);
                    return false;
                }

                return SetOption(field, value);
            }
            else
            {
                // Parse a required argument.
                if (requiredOptions.Count == 0)
                {
                    ShowError("Too many arguments");
                    return false;
                }

                FieldInfo field = requiredOptions.Peek();

                if (!IsList(field))
                {
                    requiredOptions.Dequeue();
                }

                return SetOption(field, arg);
            }
        }


        bool SetOption(FieldInfo field, string value)
        {
            try
            {
                if (IsList(field))
                {
                    // Append this value to a list of options.
                    GetList(field).Add(ChangeType(value, ListElementType(field)));
                }
                else
                {
                    // Set the value of a single option.
                    field.SetValue(optionsObject, ChangeType(value, field.FieldType));
                }

                return true;
            }
            catch
            {
                ShowError("Invalid value '{0}' for option '{1}'", value, GetOptionName(field));
                return false;
            }
        }

        static object ChangeType(string value, Type type)
        {
            TypeConverter converter = TypeDescriptor.GetConverter(type);

            return converter.ConvertFromInvariantString(value);
        }

        static bool IsList(FieldInfo field)
        {
            return typeof(IList).IsAssignableFrom(field.FieldType);
        }

        IList GetList(FieldInfo field)
        {
            return (IList)field.GetValue(optionsObject);
        }

        static Type ListElementType(FieldInfo field)
        {
            var interfaces = from i in field.FieldType.GetInterfaces()
                             where i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IEnumerable<>)
                             select i;

            return interfaces.First().GetGenericArguments()[0];
        }


        static string GetOptionName(FieldInfo field)
        {
            var nameAttribute = GetAttribute<NameAttribute>(field);

            if (nameAttribute != null)
            {
                return nameAttribute.Name;
            }
            else
            {
                return field.Name;
            }
        }

        static string GetOptionDescription(FieldInfo field)
        {
            var nameAttribute = GetAttribute<NameAttribute>(field);

            if (nameAttribute != null)
            {
                return nameAttribute.Description;
            }
            else
            {
                return "";
            }
        }

        void ShowError(string message, params object[] args)
        {
            string name = Path.GetFileNameWithoutExtension(Process.GetCurrentProcess().ProcessName);

            Console.Error.WriteLine(message, args);
            Console.Error.WriteLine();
            Console.Error.WriteLine("Usage: {0} {1}", name, string.Join(" ", requiredUsageHelp));

            if (optionalUsageHelp.Count > 0)
            {
                Console.Error.WriteLine();
                Console.Error.WriteLine("Options:");

                foreach (string optional in optionalUsageHelp)
                {
                    Console.Error.WriteLine("    {0}", optional);
                }
            }
        }

        static T GetAttribute<T>(ICustomAttributeProvider provider) where T : Attribute
        {
            return provider.GetCustomAttributes(typeof(T), false).OfType<T>().FirstOrDefault();
        }

        // Used on optionsObject fields to indicate which options are required.
        [AttributeUsage(AttributeTargets.Field)]
        public sealed class RequiredAttribute : Attribute
        {
        }

        [AttributeUsage(AttributeTargets.Field)]
        public sealed class IgnoreAttribute : Attribute
        {
        }

        // Used on an optionsObject field to rename the corresponding commandline option.
        [AttributeUsage(AttributeTargets.Field)]
        public sealed class NameAttribute : Attribute
        {
            public NameAttribute(string name)
            {
                this.Name = name;
                this.Description = "";
            }

            public NameAttribute(string name, string description)
            {
                this.Name = name;
                this.Description = description;
            }

            public string Name { get; private set; }
            public string Description { get; private set; }
        }
    }
}