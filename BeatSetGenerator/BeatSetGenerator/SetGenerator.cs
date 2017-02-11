using System;
using System.Collections.Generic;
using NAudio;
using NVorbis;
using System.IO;
using NAudio.Dsp;
using System.Diagnostics;

namespace BeatSetGenerator
{
	class SetGenerator
	{
		private float[] sampleBuffer;

		private int frameRate;
		private int sampleRate = 48000;

		public static string outputDirectory;
		private int framesPerExample;
		private BeatFileParser beatParser;
		private const int extractorCount = 1;
		private const int totalFeatures = 120;
		private FeatureExtractor[] extractors;

		public SetGenerator(int fftExponent, int frameRate, int framesPerExample)
		{
			this.framesPerExample = framesPerExample;
			extractors = new FeatureExtractor[extractorCount];
			for (int i = 0; i < extractorCount; ++i)
			{
				extractors[i] = new FeatureExtractor(100, totalFeatures / extractorCount, 10 + i, 30);
			}
			beatParser = new BeatFileParser(100, 5);
		}

		public void ConvertFile(string audioFileName, string txtFileName)
		{
			using (MonoFileReader audio = new MonoFileReader(audioFileName))
			{
				beatParser.ParseFile(txtFileName, (float)audio.TotalLength.TotalSeconds);

				long sampleCount = audio.TotalSamples;
				if (sampleBuffer == null || sampleBuffer.LongLength < sampleCount)
				{
					sampleBuffer = new float[sampleCount];
				}
				int readCount = audio.SampleProvider.Read(sampleBuffer, 0, (int)sampleCount);

				float[][,] fs = new float[extractorCount][,];
				for(int i = 0; i < extractorCount; ++i)
				{
					fs[i] = extractors[i].ExtractFeatures(sampleBuffer);
				}
				int fileCount = fs[0].GetLength(0) / framesPerExample;

				for(int file = 0; file < fileCount; ++file)
				{
					using(BinaryWriter writer = GetNextExampleFile())
					{
						for (int t = file * framesPerExample; t < (file + 1) * framesPerExample; ++t)
						{
							writer.Write(1 - beatParser.Beats[t]);
							writer.Write(beatParser.Beats[t]);
							for (int e = 0; e < extractorCount; ++e)
							{
								for(int f = 0; f < totalFeatures / extractorCount; ++f)
								{
									writer.Write(fs[e][t, f]);
								}
							}
						}
					}
				}
			}
		}
	

		static object nextExampleLock = new object();
		static private int outputFileIndex = 0;
		private string lastExampleName;
		private BinaryWriter GetNextExampleFile()
		{
			lock (nextExampleLock)
			{
				string name = outputFileIndex.ToString("D6");
				outputFileIndex++;
				name = outputDirectory + name + ".bin";
				lastExampleName = name;
				return new BinaryWriter(File.Open(name, FileMode.Create));
			}
		}
	}
}
