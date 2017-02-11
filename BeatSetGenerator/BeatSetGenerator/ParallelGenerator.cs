using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BeatSetGenerator
{
	class ParallelGenerator
	{
		private static IEnumerable<Tuple<string, string>> GetFilePairs(string dir)
		{
			string[] extensions = new[] { ".ogg", ".wav" };
			DirectoryInfo dirInfo = new DirectoryInfo(dir);
			IEnumerable<FileInfo> audioFiles =
				dirInfo.EnumerateFiles().Where(file => (extensions.Contains(file.Extension)));
			foreach (FileInfo audio in audioFiles)
			{
				string pattern = Path.GetFileNameWithoutExtension(audio.Name) + "*";
				IEnumerable<FileInfo> texts = dirInfo.EnumerateFiles(pattern).Where(file => !extensions.Contains(file.Extension));
				if (texts.Count() > 0)
				{
					yield return new Tuple<string, string>(audio.FullName, texts.First().FullName);
				}
			}
		}

		private IEnumerator<Tuple<string, string>> enumerator;

		Task[] tasks;
		public ParallelGenerator(string path, int frameRate, int exampleLength)
		{
			Action action = () =>
			{
				var gen = new SetGenerator(12, frameRate, exampleLength);
				while (true)
				{
					Tuple<string, string> item;
					lock (enumerator)
					{
						if (!enumerator.MoveNext()) break;
						item = enumerator.Current;
					}
					try
					{
						Console.WriteLine(item.Item1);
						gen.ConvertFile(item.Item1, item.Item2);
					}catch(Exception e)
					{
						Console.WriteLine("FAILED: {0}, Ex: {1}", item.Item1, e.Message);
					}
				}
			};

			enumerator = GetFilePairs(path).GetEnumerator();
			tasks = new Task[8];
			for(int i = 0; i < 8; ++i)
			{
				tasks[i] = new Task(action);
			}
		}

		public void Run()
		{
			foreach(var task in tasks)
			{
				task.Start();
			}
			foreach (var task in tasks)
			{
				task.Wait();
			}
		}

	}
}
