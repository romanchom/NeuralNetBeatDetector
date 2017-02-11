using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BeatSetGenerator
{
	class MelAggregator
	{
		public int FilterBankCount { get; private set; }
		public float[] Result { get; private set; }
		public MelAggregator( int count,float fLow, float fHigh, int fftLength, int samplingRate = 48000)
		{
			FilterBankCount = count;
			filterPoints = new int[count + 2];

			float melLow = HzToMel(fLow);
			float melHigh = HzToMel(fHigh);
			for (int i = 0; i < count + 2; ++i)
			{
				float t = (float) i / (float) (count + 1);
				float f = (1 - t) * melLow + t * melHigh;
				filterPoints[i] = (int) Math.Round(MelToHz(f) * fftLength / samplingRate);
			}

			Result = new float[count];
		}

		private int[] filterPoints;

		public void Aggregate(float[] fft)
		{
			for(int i = 0; i < FilterBankCount; ++i)
			{
				float sum = 0;
				int l = filterPoints[i];
				int c = filterPoints[i + 1];
				int r = filterPoints[i + 2];
				for(int j = l + 1; j < c; ++j)
				{
					float weight = j - l;
					weight /= (c - l);
					sum += fft[j] * weight;
				}
				for (int j = c; j < r; ++j)
				{
					float weight = r - j;
					weight /= (r - c);
					sum += fft[j] * weight;
				}
				Result[i] = (float) Math.Max(0, Math.Log(sum) + 12);
			}
		}

		private static float HzToMel(float hz)
		{
			return (float) (1127 * Math.Log(1 + hz / 700));
		}

		private static float MelToHz(float mel)
		{
			return (float) (700 * (Math.Exp(mel / 1127) - 1));
		}
	}
}
