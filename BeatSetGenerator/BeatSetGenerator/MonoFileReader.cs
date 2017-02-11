using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NAudio;
using NAudio.Wave;
using NAudio.Vorbis;
using NAudio.Wave.SampleProviders;
using System.Diagnostics;

namespace BeatSetGenerator
{
	class MonoFileReader : IDisposable
	{
		private WaveStream readerStream;
		private MediaFoundationResampler resampler;
		public ISampleProvider SampleProvider { get; private set; }

		public MonoFileReader(string fileName)
		{
			OpenFile(fileName);
			var outFormat = WaveFormat.CreateIeeeFloatWaveFormat(48000, 1);
			resampler = new MediaFoundationResampler(readerStream, outFormat);
			SampleProvider = resampler.ToSampleProvider();
			Debug.Assert(SampleProvider.WaveFormat.SampleRate == 48000);
		}

		private void OpenFile(string fileName)
		{
			if (fileName.EndsWith(".wav")) {
				readerStream = new WaveFileReader(fileName);
			}else if (fileName.EndsWith(".ogg")) {
				readerStream = new VorbisWaveReader(fileName);
			}else {
				throw new Exception("Unsupported file format.");
			}
		}

		public TimeSpan TotalLength
		{
			get
			{
				return readerStream.TotalTime;
			}
		}

		public int TotalSamples
		{
			get
			{
				return (int)(readerStream.TotalTime.TotalSeconds * 48000);
			}
		}

		#region IDisposable Support
		private bool disposedValue = false; // To detect redundant calls

		protected virtual void Dispose(bool disposing)
		{
			if (!disposedValue)
			{
				if (disposing)
				{
					resampler.Dispose();
					readerStream.Dispose();
				}
				disposedValue = true;
			}
		}

		public void Dispose()
		{
			Dispose(true);
		}
		#endregion
	}
}
