using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Text;

namespace Oxel
{
    public class PercentageConverter : TypeConverter
    {
        public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
        {
            return sourceType == typeof(string);
        }

        public override object ConvertFrom(ITypeDescriptorContext context, CultureInfo culture, object value)
        {
            try
            {
                if (value is string)
                {
                    string[] values = ((string)value).Split('%');

                    foreach (string v in values)
                    {
                        float amount;
                        if (float.TryParse(v, NumberStyles.Any, null, out amount))
                            return amount / 100.0f;
                    }

                    throw new FormatException("Not a valid percentage");
                }

                return base.ConvertFrom(context, culture, value);
            }
            catch (Exception)
            {
                return 0.0f;
            }
        }

        public override object ConvertTo(ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType)
        {
            if (destinationType == typeof(string) && value is float)
            {
                return ((float)value).ToString("P");
            }
            return base.ConvertTo(context, culture, value, destinationType);
        }
    }
}