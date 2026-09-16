"""
The real MEA2100-Mini-60, controlled through pythonnet + McsUsbNet.dll.

It has the same methods as Brian2/simulated_mea.py, so the run scripts for
the two versions are almost line-for-line the same.

Hardware notes:
  - The headstage has 2 stimulus generators ("STG 1" and "STG 2").
    We wire every left-half electrode to STG 1 and every right-half
    electrode to STG 2, so one generator = one side of the MEA.
  - Recording runs continuously in the background. The DLL calls
    _on_data() every DATA_CHUNK_MS with new samples.
"""
import threading
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

import clr
clr.AddReference(str(Path(__file__).parent / "McsUsbNet.dll"))
from System import Int32, UInt32, UInt64, Array, IntPtr
from System.Runtime.InteropServices import Marshal
from Mcs.Usb import (CMcsUsbListNet, DeviceEnumNet, CMeaUSBDeviceNet, CStg200xDownloadNet,
                     CSCUFunctionNet, DataModeEnumNet, DigitalDatastreamEnableEnumNet,
                     ChannelBlockTypeNet, SampleSizeNet, SampleDstSizeNet,
                     ElectrodeModeEnumNet, ElectrodeDacMuxEnumNet, STG_DestinationEnumNet)

from feedback import N_CHANNELS, LEFT_CHANNELS

HEADSTAGE = 0          # SCU port the headstage is plugged into (HS1 = 0, HS2 = 1, ...)
SAMPLE_RATE = 50_000   # Hz
DATA_CHUNK_MS = 20     # how often the DLL hands us new samples

# Stimulation defaults, taken from DishBrain (Kagan et al., 2022).
# Check these with the lab lead before running on a live culture.
STIM_AMPLITUDE_UV = 75_000  # 75 mV biphasic pulses
PHASE_US = 200              # length of each pulse phase (microseconds)
PULSES_PER_WINDOW = 2       # pulses per stimulation window (same count for both kinds)

SPIKE_THRESHOLD_SD = 5      # a spike is a dip below -5 x noise level


def count_spikes(voltages, sample_rate=SAMPLE_RATE):
    """
    voltages: array of shape (channels, samples).
    Returns the number of spikes on each channel.

    1. Band-pass 300-3000 Hz to keep only the spike frequencies.
    2. Estimate each channel's noise level: median(|signal|) / 0.6745.
    3. Count how many times the signal drops below -SPIKE_THRESHOLD_SD x noise.
    """
    sos = butter(2, [300, 3000], btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, voltages, axis=1)
    noise = np.median(np.abs(filtered), axis=1) / 0.6745
    below = filtered < -SPIKE_THRESHOLD_SD * noise[:, None]
    new_crossings = below[:, 1:] & ~below[:, :-1]
    return new_crossings.sum(axis=1)


def to_numpy(net_array, length):
    """Fast copy of a .NET Int32[] into a numpy array."""
    out = np.empty(length, dtype=np.int32)
    Marshal.Copy(net_array, 0, IntPtr(out.ctypes.data), length)
    return out


class MEADevice:
    def __init__(self):
        self.dacq = None  # recording side of the device
        self.stg = None   # stimulation side of the device
        self._chunks = []
        self._lock = threading.Lock()
        self._frames_per_chunk = SAMPLE_RATE * DATA_CHUNK_MS // 1000

    # ---------- setup ----------

    def connect(self):
        devices = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
        if devices.Count == 0:
            raise RuntimeError("No MEA found. Is it plugged in and switched on?")
        entry = devices.GetUsbListEntry(0)

        self.dacq = CMeaUSBDeviceNet()
        self.dacq.ChannelDataEvent += self._on_data
        self.dacq.ErrorEvent += lambda msg, action: print("MEA error:", msg)
        status = self.dacq.Connect(entry)
        if status != 0:
            raise RuntimeError(f"Could not connect to MEA (status {status})")

        self.stg = CStg200xDownloadNet()
        self.stg.Connect(entry)

        self._setup_recording()
        self._setup_stimulation()
        self.dacq.StartDacq()
        print("MEA connected and recording.")

    def _setup_recording(self):
        # Follows MCS's MEA2100_Recording_and_Stimulation example.
        self.dacq.StopDacq(0)
        CSCUFunctionNet(self.dacq).SetDacqLegacyMode(False)
        self.dacq.SetSamplerate(SAMPLE_RATE, 0, 0)
        self.dacq.SetDataMode(DataModeEnumNet.Signed_32bit, 0)
        self.dacq.SetNumberOfAnalogChannels(60, 0, 0, 8, 0)
        self.dacq.EnableDigitalIn(DigitalDatastreamEnableEnumNet.DigitalIn
                                  | DigitalDatastreamEnableEnumNet.DigitalOut
                                  | DigitalDatastreamEnableEnumNet.Hs1SidebandLow
                                  | DigitalDatastreamEnableEnumNet.Hs1SidebandHigh, 0)
        self.dacq.EnableChecksum(True, 0)

        _, _, checksum_ch, timestamp_ch, channels_in_block = self.dacq.GetChannelLayout(
            Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), 0)
        n_streams = channels_in_block // 2  # 32-bit data uses 2 words per stream

        block = self.dacq.ChannelBlock
        block.Init(channels_in_block)
        block.AddBlocksAndChannels(ChannelBlockTypeNet.MultipleHandlesOneQueues,
                                   Array[bool]([True] * n_streams), SAMPLE_RATE,
                                   self._frames_per_chunk, SampleSizeNet.SampleSize32Signed,
                                   SampleDstSizeNet.SampleDstSize32, 0, 0)
        block.SetCommonThreshold(self._frames_per_chunk)
        block.SetCheckChecksum(checksum_ch, timestamp_ch)

    def _setup_stimulation(self):
        for electrode in range(N_CHANNELS):
            generator = ElectrodeDacMuxEnumNet.Stg1 if electrode in LEFT_CHANNELS else ElectrodeDacMuxEnumNet.Stg2
            self.stg.SetElectrodeMode(HEADSTAGE, electrode, ElectrodeModeEnumNet.emAutomatic)
            self.stg.SetElectrodeEnable(HEADSTAGE, electrode, 0, True)
            self.stg.SetElectrodeDacMux(HEADSTAGE, electrode, 0, generator)
            self.stg.SetEnableAmplifierProtectionSwitch(HEADSTAGE, electrode, False)
            self.stg.SetBlankingEnable(HEADSTAGE, electrode, True)  # hide stim artifacts
        self.stg.SetVoltageMode(0)
        self.stg.SetupTrigger(0, Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([1]))

    # ---------- recording ----------

    def _on_data(self, dacq, handle, num_frames):
        """Called by the DLL on its own thread whenever a chunk of samples is ready."""
        chunk = np.empty((N_CHANNELS, self._frames_per_chunk), dtype=np.int32)
        for channel in range(N_CHANNELS):
            data, n = dacq.ChannelBlock.ReadFramesI32(channel, 0, self._frames_per_chunk, Int32(0))
            chunk[channel, :n] = to_numpy(data, n)
            chunk[channel, n:] = 0
        with self._lock:
            self._chunks.append(chunk)

    def read_spike_counts(self, window_ms):
        """Record for window_ms and return spikes per channel (length 60)."""
        with self._lock:
            self._chunks.clear()
        time.sleep(window_ms / 1000)
        needed = SAMPLE_RATE * window_ms // 1000
        give_up_at = time.time() + 2
        while sum(c.shape[1] for c in self._chunks) < needed:
            if time.time() > give_up_at:
                raise RuntimeError("No data from the MEA for 2 s. Check the USB connection.")
            time.sleep(0.005)
        with self._lock:
            voltages = np.concatenate(self._chunks, axis=1)[:, :needed]
        return count_spikes(voltages.astype(float))

    # ---------- stimulation ----------

    def stimulate_predictable(self, side, window_ms=100):
        """Structured feedback: evenly spaced pulses on the side the pole leans toward."""
        spacing_us = window_ms * 1000 // PULSES_PER_WINDOW
        onsets = [i * spacing_us for i in range(PULSES_PER_WINDOW)]
        pulses = self._pulse_train(onsets, window_ms)
        silent = self._pulse_train([], window_ms)
        if side == "left":
            self._send(pulses, silent)
        else:
            self._send(silent, pulses)

    def stimulate_unpredictable(self, window_ms=100):
        """Unstructured feedback: the same number of pulses at random times, on both sides."""
        pulse_us = 2 * PHASE_US
        n_slots = window_ms * 1000 // pulse_us  # possible pulse start times, no overlaps

        def random_onsets():
            return sorted(int(slot) * pulse_us for slot in np.random.choice(n_slots, PULSES_PER_WINDOW, replace=False))

        self._send(self._pulse_train(random_onsets(), window_ms), self._pulse_train(random_onsets(), window_ms))

    def _pulse_train(self, onsets_us, window_ms):
        """
        Build one generator's waveform as (amplitudes in uV, durations in us, stim-active flags).
        Each pulse is +amplitude then -amplitude (biphasic), with silence in between.
        """
        amplitudes, durations, active = [], [], []
        t = 0
        for onset in onsets_us:
            if onset > t:
                amplitudes.append(0); durations.append(onset - t); active.append(0)
            amplitudes += [STIM_AMPLITUDE_UV, -STIM_AMPLITUDE_UV]
            durations += [PHASE_US, PHASE_US]
            active += [1, 1]
            t = onset + 2 * PHASE_US
        end = window_ms * 1000
        if end > t:
            amplitudes.append(0); durations.append(end - t); active.append(0)
        return amplitudes, durations, active

    def _send(self, stg1_train, stg2_train):
        for generator, (amplitudes, durations, active) in enumerate([stg1_train, stg2_train]):
            channel = 2 * HEADSTAGE + generator
            amp = Array[Int32](amplitudes)
            dur = Array[UInt64](durations)
            # The sideband tells the headstage when to connect the electrodes to the generator.
            sideband = self.stg.Stimulus.CreateSideband(Array[Int32](active), Array[Int32]([0] * len(active)),
                                                        dur, 40, 800, 40)
            self.stg.PrepareAndSendData(channel, amp, dur, STG_DestinationEnumNet.channeldata_voltage)
            self.stg.PrepareAndSendData(channel, sideband.Sideband, sideband.Duration,
                                        STG_DestinationEnumNet.syncoutdata)
        self.stg.SendStart(1)

    # ---------- shutdown ----------

    def close(self):
        if self.dacq is not None:
            self.dacq.StopDacq(0)
            self.dacq.Disconnect()
        if self.stg is not None:
            self.stg.Disconnect()
        print("MEA disconnected.")
