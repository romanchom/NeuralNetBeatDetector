using System;

namespace BeatSetGenerator
{
	class ToneAggregator
	{
		private readonly int toneCount;
		private readonly float lowFrequency;
		private readonly float octaveCount;
		private readonly float step;
		private float fMul;
		private readonly float logMul;

		public float[] Tones
		{
			get; private set;
		}

		public int SamplingRate
		{
			set
			{
				fMul = (float)windowSize / value;
			}
		}

		private int windowSize;

		public ToneAggregator(int toneCount, float lowFrequency, float highFrequency, int samplingRate, int windowSize, float logarithmBase = 10, float unitsPerDecade = 2)
		{
			this.toneCount = toneCount;
			this.lowFrequency = lowFrequency;
			this.windowSize = windowSize;

			logMul = (float)(0.5 / Math.Log(logarithmBase)) * unitsPerDecade;

			octaveCount = (float)(Math.Log(highFrequency / lowFrequency) / Math.Log(2));
			step = octaveCount / toneCount;

			SamplingRate = samplingRate;

			Tones = new float[toneCount];
		}

		public void Aggregate(float[] magnitudes, float norm)
		{
			int jl = (int)(lowFrequency * fMul);
			for (int i = 0; i < toneCount; ++i)
			{
				float f1 = lowFrequency * (float)Math.Pow(2, step * (i + 1));
				int jh = (int)(f1 * fMul);
				if (jh <= jl) jh = jl + 1;

				float sum = 0;
				for (int j = jl; j < jh; ++j)
				{
					sum += magnitudes[j];
				}
				Tones[i] = (float)Math.Log(sum) * logMul;
				norm = Math.Max(norm, Tones[i]);

				jl = jh;
			}
			for (int i = 0; i < toneCount; ++i)
			{
				Tones[i] -= norm;
			}
		}
	}
}
