using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BeatSetGenerator
{
	class BeatFileParser
	{
		public float[] Beats { get; private set; }

		private int dataSize;
		private float[] raw;
		private float[] kernel;
		private float frequency;

		public BeatFileParser(float frequency, int kernelSize)
		{
			this.frequency = frequency;
			kernel = new float[kernelSize];
			Func<float, float> Cosine = (x => (float)(1 + Math.Cos(x * Math.PI) / 2));
			Func<float, float> Gauss = (x => (float)Math.Exp(-x * x * 3));

			for (int i = 0; i < kernelSize; ++i)
			{
				float x = (float)(i + 1) / (float)(kernelSize + 1);
				x = x * 2 - 1;
				kernel[i] = Gauss(x);
			}
		}

		public void ParseFile(string name, float timeLength)
		{
			PrepareBuffers(timeLength);
			using(StreamReader reader = new StreamReader(name))
			{
				string line = reader.ReadLine();
				reader.BaseStream.Seek(0, SeekOrigin.Begin);
				reader.DiscardBufferedData();

				if (name.EndsWith(".beats")) {
					ParseBallroom(reader);
				} else if (line.Contains(",")) {
					ParseNecroDancerInt(reader);
				} else if (line.Contains("\t")) {
					ParseMirex(reader);
				} else {
					ParseNecroDancerFloat(reader);
				}
			}

			ConvolveWithKernel();
		}

		private void PrepareBuffers(float timeLength)
		{
			dataSize = (int)(timeLength * frequency) + 1;
			if (raw == null || raw.Length < dataSize)
			{
				raw = new float[dataSize];
				Beats = new float[dataSize];
			}else
			{
				for(int i = 0; i < dataSize; ++i)
				{
					raw[i] = 0;
					Beats[i] = 0;
				}
			}
		}

		private void ParseNecroDancerInt(StreamReader reader)
		{
			int[] beatPositions = Array.ConvertAll(reader.ReadToEnd().Split(','), s => (int)(int.Parse(s) * frequency / 1000.0f));
			foreach(int i in beatPositions)
			{
				raw[i] = 1;
			}
		}

		private void ParseNecroDancerFloat(StreamReader reader)
		{
			string line;
			while(!string.IsNullOrEmpty(line = reader.ReadLine()))
			{
				ParseBeatTimeInSeconds(line);
			}
		}

		private void ParseMirex(StreamReader reader)
		{
			string line;
			while (!string.IsNullOrEmpty(line = reader.ReadLine()))
			{
				foreach(string word in line.Split(' ', '\t'))
				{
					ParseBeatTimeInSeconds(word);
				}
			}
		}

		private void ParseBallroom(StreamReader reader)
		{
			string line;
			while (!string.IsNullOrEmpty(line = reader.ReadLine()))
			{
				string word = line.Split(' ')[0];
				ParseBeatTimeInSeconds(word);
			}
		}

		private void ParseBeatTimeInSeconds(string word)
		{
			float s;
			if (float.TryParse(word, out s))
			{
				int index = (int)(frequency * s);
				raw[index] += 1;
			}
		}

		private void ConvolveWithKernel()
		{
			int k2 = kernel.Length / 2;
			for (int i = 0; i < dataSize; ++i)
			{
				if (raw[i] > 0)
				{
					int center = i;
					int low = i - k2;
					int high = i + k2 + 1;
					int j = Math.Max(0, low) - low;
					int end = Math.Min(dataSize, high) - low;
					for (; j < end; ++j) {
						Beats[i + j - k2] += kernel[j] * raw[i];
					}
				}
			}
			float max = 0;
			for (int i = 0; i < dataSize; ++i)
			{
				max = Math.Max(max, Beats[i]);
			}

			max = max * 0.5f + 1e-10f;

			for (int i = 0; i < dataSize; ++i)
			{
				float val = Math.Min(1, Beats[i] / max);
				Beats[i] = val;
			}
		}
	}
}
