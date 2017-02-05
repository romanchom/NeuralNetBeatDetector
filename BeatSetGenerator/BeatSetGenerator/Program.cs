using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace BeatSetGenerator
{
	class Program
	{
		static void Main(string[] args)
		{
			SetGenerator gen = new SetGenerator(10, 100, 200);
			gen.ConvertDirectory(args[0], args[1]);
		}
	}
}
