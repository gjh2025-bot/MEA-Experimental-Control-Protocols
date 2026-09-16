# Baseline Protocol

The simplest working version of the Neurotech@Berkeley MEA project. Neurons on a
microelectrode array (MEA) play CartPole: we tell them which way the pole is
leaning by stimulating them, read their spikes, and turn those spikes into
left/right moves.

There are two versions that follow the same steps:

| Folder      | Neurons                          | Needs hardware?          |
|-------------|----------------------------------|--------------------------|
| `Brian2/`   | 60 simulated neurons (Brian2)    | No. Start here.          |
| `Hardware/` | Real culture on the MEA2100-Mini-60 | Yes, over USB (Windows) |

## The loop

```
             ┌───────────────────────────────────────────────┐
             ▼                                               │
   CartPole pole angle                                       │
             │                                               │
   balanced? (|angle| < 0.15 rad)                            │
     yes → predictable stim on the side the pole leans to    │
     no  → unpredictable stim (same amount, random)          │
             │                                               │
   record spikes for 100 ms while the stim plays             │
             │                                               │
   more spikes on left half → push left (0)                  │
   more spikes on right half → push right (1)                │
             │                                               │
   env.step(action) ─────────────────────────────────────────┘
```

The idea comes from DishBrain (Kagan et al., 2022): cultured neurons get
predictable input when things go well and unpredictable input when they don't.
The hypothesis is that the network reorganizes to make its input more predictable.

## Files

Each folder has the same three files:

| File | What it does |
|------|--------------|
| `feedback.py` | The rules: when the pole counts as balanced, which side to stimulate, how spikes become an action. **Identical in both folders.** |
| `simulated_mea.py` / `mea_device.py` | The "MEA". Both have the same methods: `connect()`, `stimulate_predictable(side)`, `stimulate_unpredictable()`, `read_spike_counts(window_ms)`, `close()`. |
| `run_brian2.py` / `run_hardware.py` | The game loop. Compare the two and you'll see they are nearly the same. |

## Setup

```bash
pip install -r requirements.txt
```

## Running

**Brian2** (any computer, about 5 s per episode):

```bash
cd Brian2
python run_brian2.py
```

You should see episodes lasting about 25–45 steps. A random policy averages about 22.
The simulation wires the lean-side neurons straight to the matching action, so it
beats random from the first episode. It is not expected to improve over episodes.

**Hardware** (lab computer with the MEA plugged in):

```bash
cd Hardware
python run_hardware.py
```

Before the first run on a live culture:
- `McsUsbNet.dll` must sit next to `mea_device.py`.
- Set `HEADSTAGE` in `mea_device.py` to the SCU port the headstage is plugged into (HS1 = 0).
- Check `STIM_AMPLITUDE_UV`, `PHASE_US`, and `PULSES_PER_WINDOW` with the lab lead.
  The defaults (75 mV, 200 µs per phase) are DishBrain's values.
- Do a dry run with the MCS test model probe (Test-60MEA) before using cells.

Both scripts save `episode_lengths.csv` and show a plot at the end.

## Things to try

- Change `BALANCE_LIMIT_RAD` in `feedback.py`. Remember to change it in both folders.
- Brian2: change `STIM_CURRENT`, or the background rate in `simulated_mea.py`, and see how episode length changes.
- Brian2: print `mea.synapses.w` before and after a run to watch STDP change the weights.

## What the full codebase adds

The full versions in `Brian2-Path/` and `Hardware-Pythonnet-Path/` (repo root) build on this baseline with:
- Lateral inhibition between the two halves
- Homeostatic synaptic scaling
- A reward trace that modulates STDP
- A 3 s noise burst when an episode ends
- Angle/velocity-encoded stimulation waveforms (`square.py`)
- A filter-based spike detector (`spike.py`)
