using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BeatSetGenerator
{
	class Program
	{
		static void Main(string[] args)
		{
			try
			{
				int frameRate = int.Parse(args[2]);
				int length = int.Parse(args[3]);
				SetGenerator.outputDirectory = args[1];
				ParallelGenerator gen = new ParallelGenerator(args[0], frameRate, length);
				gen.Run();

			}
			catch (Exception e)
			{
				Console.WriteLine(e.Message);
			}
		}
	}
}
