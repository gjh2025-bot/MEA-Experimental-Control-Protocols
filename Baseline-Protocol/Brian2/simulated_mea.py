"""
A simulated MEA: 60 spiking neurons in Brian2, one neuron per "electrode".

It has the same methods as Hardware/mea_device.py, so the run scripts for
the two versions are almost line-for-line the same.
"""
import numpy as np
from brian2 import (NeuronGroup, Synapses, PoissonInput, SpikeMonitor,
                    Network, defaultclock, prefs, ms, Hz)

from feedback import N_CHANNELS, LEFT_CHANNELS, RIGHT_CHANNELS

prefs.codegen.target = "numpy"  # no C++ compiler needed

STIM_CURRENT = 10.0  # input current for a stimulated neuron (Izhikevich units)
W_MAX = 5.0          # largest allowed synapse weight (mV jump per spike)


class SimulatedMEA:
    def __init__(self):
        defaultclock.dt = 0.1 * ms

        # Izhikevich "regular spiking" neurons. v = membrane voltage (mV),
        # u = recovery variable, I = input current set by stimulation.
        self.neurons = NeuronGroup(
            N_CHANNELS,
            """
            dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms : 1
            du/dt = 0.02 * (0.2*v - u) / ms : 1
            I : 1
            """,
            threshold="v >= 30",
            reset="v = -65; u += 8",
            method="euler",
        )
        self.neurons.v = -65
        self.neurons.u = -13

        # Random excitatory connections that learn with STDP
        # (spike-timing-dependent plasticity):
        #   pre fires shortly before post -> weight goes up   (LTP)
        #   post fires shortly before pre -> weight goes down (LTD)
        self.synapses = Synapses(
            self.neurons, self.neurons,
            model="""
            w : 1
            dApre/dt  = -Apre  / (20*ms) : 1 (event-driven)
            dApost/dt = -Apost / (20*ms) : 1 (event-driven)
            """,
            on_pre="""
            v_post += w
            Apre += 0.01 * W_MAX
            w = clip(w + Apost, 0, W_MAX)
            """,
            on_post="""
            Apost -= 0.0105 * W_MAX
            w = clip(w + Apre, 0, W_MAX)
            """,
            namespace={"W_MAX": W_MAX},
        )
        self.synapses.connect(condition="i != j", p=0.1)
        self.synapses.w = "rand() * 2"

        # Random background input so the network fires spontaneously
        # (~7 Hz per neuron), like a real culture does.
        background = PoissonInput(self.neurons, "v", N=100, rate=20 * Hz, weight=1.5)

        self.spikes = SpikeMonitor(self.neurons, record=False)  # count spikes only
        self.network = Network(self.neurons, self.synapses, background, self.spikes)

    def connect(self):
        print("Simulated MEA ready (no hardware).")

    def stimulate_predictable(self, side):
        """Structured feedback: stimulate every neuron on the side the pole leans toward."""
        channels = LEFT_CHANNELS if side == "left" else RIGHT_CHANNELS
        self.neurons.I = 0
        self.neurons.I[channels] = STIM_CURRENT

    def stimulate_unpredictable(self):
        """Unstructured feedback: the same amount of input, sent to random neurons."""
        random_channels = np.random.choice(N_CHANNELS, size=len(LEFT_CHANNELS), replace=False)
        self.neurons.I = 0
        self.neurons.I[random_channels] = STIM_CURRENT

    def read_spike_counts(self, window_ms):
        """Run the network for window_ms and return spikes per channel (length 60)."""
        before = np.array(self.spikes.count)
        self.network.run(window_ms * ms)
        self.neurons.I = 0  # stimulation only lasts for this window
        return np.array(self.spikes.count) - before

    def close(self):
        pass
