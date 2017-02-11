using NAudio.Dsp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BeatSetGenerator
{
	class FeatureExtractor
	{
		public int Frequency { get; private set; }
		public int FeatureCount { get; private set; }
		public int WindowExponent { get; private set; }
		private int MedianLength { get; set; }

		public FeatureExtractor(int frequency = 100, int featureCount = 120, int windowExponent = 10, int medianLength = 20)
		{
			Frequency = frequency;
			FeatureCount = featureCount;
			WindowExponent = windowExponent;
			MedianLength = medianLength;

			fftLength = 1 << WindowExponent;
			fftBuffer = new Complex[fftLength];
			window = new float[fftLength];
			magnitudes = new float[fftLength];

			for (int i = 0; i < fftLength; ++i)
			{
				window[i] = (float)FastFourierTransform.BlackmannHarrisWindow(i, fftLength);
			}
			aggregator = new MelAggregator(featureCount / 2, 100, 12000, fftLength, sampleRate);
		}

		const int sampleRate = 48000;
		private int fftLength;
		private Complex[] fftBuffer;
		private float[] window;
		private float[] magnitudes;
		private MelAggregator aggregator;

		public float[,] ExtractFeatures(float[] samples)
		{
			int length = samples.Length * Frequency / sampleRate;
			float[,] features = new float[length, FeatureCount];

			for(int i = 0; i < length; ++i)
			{
				int windowEnd = i * sampleRate / Frequency;
				int windowBegin = Math.Max(0, windowEnd - fftLength);
				for (int j = 0; j < fftLength; ++j)
				{
					int index = Math.Max(0, j + windowBegin);
					fftBuffer[j].X = samples[index] * window[j];
					fftBuffer[j].Y = 0;
				}
				FastFourierTransform.FFT(true, WindowExponent, fftBuffer);
				for (int j = 0; j < fftLength; ++j)
				{
					magnitudes[j] = (float)(fftBuffer[j].X * fftBuffer[j].X + fftBuffer[j].Y * fftBuffer[j].Y);
				}
				aggregator.Aggregate(magnitudes);

				for(int j = 0; j < FeatureCount/ 2; ++j)
				{
					features[i, j] = aggregator.Result[j];
				}
			}

			float[] temp0 = new float[MedianLength];
			float[] temp1 = new float[MedianLength];
			for (int f = 0; f < FeatureCount / 2; ++f)
			{
				Array.Clear(temp0, 0, MedianLength);
				for(int t = 0; t < length; ++t)
				{
					float val = features[t, f];
					temp0[t % MedianLength] = val;
					temp0.CopyTo(temp1, 0);
					Array.Sort(temp1);
					float median = temp1[MedianLength / 2];

					features[t, f + FeatureCount / 2] = Math.Max(0, val - median);
				}
			}

			return features;
		}
	}
}
