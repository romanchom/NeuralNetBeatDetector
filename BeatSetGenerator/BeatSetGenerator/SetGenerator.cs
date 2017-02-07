using System;
using System.Collections.Generic;
using NAudio;
using NVorbis;
using System.IO;
using NAudio.Dsp;

namespace BeatSetGenerator
{
	class SetGenerator
	{
		private int fftLength;
		private Complex[] fftBuffer;
		private float[] window;

		private float[] sampleBuffer;

		private float[] magnitudes;

		private int frameRate;
		private int fftExponent;
		private int samplingRate = 48000;

		private ToneAggregator aggregator;
		private int outputFileIndex = 0;
		private string outputDirectory;
		private int framesPerExample;



		public SetGenerator(int fftExponent, int frameRate, int framesPerExample)
		{
			this.fftExponent = fftExponent;
			this.framesPerExample = framesPerExample;
			fftLength = 1 << fftExponent;
			this.frameRate = frameRate;
			fftBuffer = new Complex[fftLength];
			window = new float[fftLength];
			magnitudes = new float[fftLength];
			for (int i = 0; i < fftLength; ++i)
			{
				window[i] = (float) FastFourierTransform.BlackmannHarrisWindow(i, fftLength);
			}

			aggregator = new ToneAggregator(120, 100, 12000, samplingRate, fftLength);
		}

		public void ConvertDirectory(string inputDir, string outputDir)
		{
			outputDirectory = outputDir;
			DirectoryInfo input = new DirectoryInfo(inputDir);
			IEnumerable<FileInfo> beatFiles = input.EnumerateFiles("*.txt");
			foreach(FileInfo beatFile in beatFiles)
			{
				string beatPath = beatFile.FullName;
				string pattern = Path.GetFileNameWithoutExtension(beatFile.Name) + ".ogg";
				IEnumerable<FileInfo> audioFiles = input.EnumerateFiles(pattern);
				foreach(FileInfo audio in audioFiles)
				{
					Console.WriteLine(audio.FullName + " " + beatFile.FullName);
			
						OpenFile(audio.FullName, beatFile.FullName);

				}
			}
		}

		private void OpenFile(string oggFileName, string txtFileName)
		{
			using (StreamReader txt = new StreamReader(txtFileName))
			using (VorbisReader ogg = new VorbisReader(oggFileName))
			{
				BinaryWriter output = GetNextExampleFile();
				int framesInExample = 0;
				aggregator.SamplingRate = samplingRate = ogg.SampleRate;

				// read entire ogg file at once
				long sampleCount = ogg.TotalSamples;
				if (sampleBuffer == null || sampleBuffer.LongLength < sampleCount)
				{
					sampleBuffer = new float[sampleCount];
				}
				ogg.ReadSamples(sampleBuffer, 0, (int)sampleCount);

				// read entire txt file at once
				int[] beatPositions;
				string contents = txt.ReadLine();
				if(txt.EndOfStream)
				{
					float msToIndex = ogg.SampleRate / 1000.0f;
					beatPositions = Array.ConvertAll(contents.Split(','), s => (int) (int.Parse(s) * msToIndex));
				}
				else
				{
					txt.BaseStream.Seek(0, SeekOrigin.Begin);
					txt.DiscardBufferedData();
					contents = txt.ReadToEnd();
					beatPositions = Array.ConvertAll(contents.Split('\n'), s =>
					{
						float ret = float.MaxValue;
						float.TryParse(s, out ret);
						return (int)(ret * samplingRate);
					});
				}
				

				// downmix to mono
				sampleCount /= 2;
				for (int i = 0; i < sampleCount; ++i)
				{
					sampleBuffer[i] = (sampleBuffer[i * 2] + sampleBuffer[i * 2 + 1]) / 2;
				}

				int windowIndex = 0;
				float[] last = new float[60];
				int lastBeatIndex = 0;
				int beatState = 0;
				const int beatLength = 5;
				while (true)
				{
					int windowEnd = windowIndex * ogg.SampleRate / frameRate;
					if (windowEnd > sampleCount) break;
					int windowBegin = windowEnd - fftLength;

					for (int i = 0; i < fftLength; ++i)
					{
						int index = Math.Max(0, i + windowBegin);
						fftBuffer[i].X = sampleBuffer[index] * window[i];
						fftBuffer[i].Y = 0;
					}

					FastFourierTransform.FFT(true, fftExponent, fftBuffer);

					for (int i = 0; i < fftLength; ++i)
					{
						magnitudes[i] = (float)(fftBuffer[i].X * fftBuffer[i].X + fftBuffer[i].Y * fftBuffer[i].Y) + 1e-15f;
					}

					//aggregator.Tones.CopyTo(last, 0);
					aggregator.Aggregate(magnitudes, 0);

					// find if current segment has beat
					float hasBeat = 0;
					if(lastBeatIndex < beatPositions.Length)
					{
						int pos = beatPositions[lastBeatIndex];
						if(pos <= windowEnd)
						{
							beatState = beatLength;
							lastBeatIndex++;
						}
					}
					hasBeat = beatState > 0 ? 1.0f : 0.0f;
					beatState--;

					output.Write(1 - hasBeat);
					output.Write(hasBeat);
					for(int i = 0; i < 120; ++i)
					{
						output.Write(aggregator.Tones[i]);
						//float diff = aggregator.Tones[i] - last[i];
						//output.Write(diff);
					}
					++framesInExample;
					if(framesInExample >= framesPerExample)
					{
						output.Dispose();
						output = GetNextExampleFile();
						framesInExample = 0;
					}					

					++windowIndex;
				}
				output.Dispose();
				if(framesInExample != 0)
				{
					File.Delete(lastExampleName);
				}
			}
		}

		private string lastExampleName;
		private BinaryWriter GetNextExampleFile()
		{
			string name = outputFileIndex.ToString("D6");
			outputFileIndex++;
			name = outputDirectory + name + ".bin";
			lastExampleName = name;
			return new BinaryWriter(File.Open(name, FileMode.Create));
		}
	}
}
