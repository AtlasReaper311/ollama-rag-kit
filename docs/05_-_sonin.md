Case Study

# Building an Autonomous Generative Audio-Visual System

SONIN: Dual-Engine Generative System

A full technical breakdown of SONIN, an autonomous digital instrument built in Max/MSP that composes its own evolving music and visuals in real time by continuously listening and reacting to the sounds it creates. This covers the dual-engine architecture, the three engineering problems that nearly broke it, and the DSP solutions that made it stable.

## Overview

Most generative music systems fall into one of two failure modes: they either produce unpredictable noise with no musical coherence, or they produce something so constrained it sounds like a sequencer on shuffle. SONIN was built to solve both problems simultaneously.

The central principle is controlled randomness. The system uses probability logic, scale constraints, and an internal audio feedback loop to ensure that its output remains structurally and tonally intentional without requiring constant human intervention. It listens to what it has just made, analyses it, and uses that analysis to decide what to make next.

Built in Max/MSP across September to December 2025, SONIN was submitted as a university project and received 100% for technical complexity and design maturity. It is also a system I intend to continue developing; the architecture is modular enough that the engines can be extended considerably.

## System Architecture

SONIN operates on a dual-engine architecture where two distinct sound generation systems run in parallel, interact through a shared feedback network, and are controlled via a unified UI layer. The four components are the Melodic Engine, the Granular Engine, the Audio Analysis and Feedback Network, and the Visual Engine.

The system's intelligence comes from the feedback network that connects all four components. Rather than operating as isolated modules, each engine's output becomes another engine's input. The signal flows in a loop: the granular engine generates texture, the analysis network reads that texture, the melodic engine reacts to the analysis, and the combined output feeds back into the analysis network again.

## The Melodic Engine

The melodic engine is responsible for the system's structural musicality. Rather than sequencing hard-coded notes, it uses a probability-driven logic gate to generate rhythmic and melodic data in real time.

### Scale-constrained pitch generation

Raw random integer generation (random 127) produces atonal chaos with no musical coherence. To resolve this, raw integers are routed through modulo arithmetic and cross-referenced against a coll object storing specific MIDI arrays (Dorian, pentatonic, or custom scales depending on the preset). The result is infinite melodic variation that never leaves the key. Changing the scale stored in the coll object instantly shifts the entire tonal character of the system without touching any other parameter.

### Emergent register shifts

By modulating the range of the random object based on the real-time audio analysis data, the melody naturally shifts registers as the system's overall density increases. When the granular engine swells and raises the master RMS, the melodic engine begins generating pitch values in a higher octave range, producing an emergent behaviour that sounds intentional without being explicitly programmed.

### Rhythmic density

Note density is controlled using decide and probability-weighted random objects acting as logic gates; each incoming clock pulse has a defined probability of triggering a note or resting. This probability is itself a parameter driven by the feedback network, meaning the rhythmic density of the melody responds to what the system is currently producing texturally.

## The Granular Engine

The granular engine acts as the textural counterweight to the melodic engine. Where the melodic engine produces structured, pitch-defined output, the granular engine produces evolving atmospheric density through micro-manipulation of audio buffers.

### Voice management with poly~

Granular synthesis requires triggering dozens of overlapping audio snippets per second. Managing this polyphony efficiently is critical; without careful voice allocation the CPU overhead becomes unmanageable and audio quality degrades. Each grain's lifecycle is encapsulated within a poly~ object, which handles voice stealing and CPU allocation. The grain size, pitch, and position within the source buffer are all independently randomisable parameters, giving the engine its characteristic evolving texture.

### Windowing and envelope management

Without proper windowing, every grain trigger produces an audible click as the audio cuts in and out abruptly. Each grain's read pointer (a phasor~) is multiplied by an amplitude envelope reading a secondary buffer~ containing a Hann window shape. This ensures every grain fades in and out smoothly regardless of where in the source buffer the read pointer starts.

### Decoupling pitch and duration

Standard sample playback links speed and pitch: playing a sample twice as fast halves its duration and raises pitch by an octave. In a granular context this hard-linked relationship severely limits textural possibilities. To decouple them, the frequency of the phasor~ driving grain generation (duration) is separated from the step-size of the buffer read pointer (pitch). A desired pitch shift calculates a rate multiplier for the read pointer while keeping the grain envelope frequency static, requiring precise expr math to maintain phase alignment and prevent artifacts.

## The Feedback Loop

The feedback network is what makes SONIN feel like a living system rather than a static playback device. The system listens to its own output and routes that analysis back into its own generative parameters.

Two values are extracted from the master output bus in real time. RMS amplitude tracks the overall energy of the system using average~. Spectral brightness is derived using zerox~ for zero-crossing rate analysis (a computationally cheap proxy for brightness) or pfft~ for full spectral centroid analysis when CPU budget allows.

Audio-rate signals are converted to control-rate floats using snapshot~, then pushed through scale and zmap objects to map them to usable parameter ranges. clip objects are applied strictly at every stage to prevent the feedback loop from pushing the system into mathematically invalid states. The scaled values then feed directly into the melodic engine's tempo, density, and pitch range parameters.

Without the feedback loop, SONIN is two independent synthesisers running in parallel. With it, the system has genuine cause-and-effect relationships between its components. A swell in the granular texture increases the RMS, which increases melodic note density, which adds more harmonic content to the output, which further modifies the spectral analysis, which shifts the melodic register. The system composes itself.

The first serious problem appeared when dynamic tempo modulation was introduced. Driving the interval of a metro object from a continuous control signal caused the system to lose its musical grid, producing erratic MIDI clustering and skipped notes.

When a metro object's interval updates mid-tick, the new value fires immediately rather than waiting for the current cycle to complete. A command to fire at 100ms arriving while the clock is 150ms into a 200ms cycle triggers an instant, unquantised pulse. Applied continuously across a dynamic control signal, this destroys rhythmic coherence entirely.

The solution was to decouple the raw control data from the clock entirely. A quantised update system was implemented using a master pulse and snapshot~ / sample-and-hold (sah~) logic. Rather than feeding continuous data directly into the clock speed, the generative data is buffered and held. The system only reads and applies the new tempo or density value at the precise moment a musical bar or subdivision ends. This preserved structural integrity while still allowing for extreme tempo manipulation driven by the audio analysis data.

Sample-and-hold matrix on the control path. New tempo values are only applied at quantised musical boundaries, not on continuous data updates. The clock stays coherent; the system stays musical.

High grain-density events produced persistent high-frequency audio clicks. The problem and the obvious fix were in direct conflict with each other.

When all allocated voices in the poly~ object are occupied and a new grain triggers, the system executes voice stealing; abruptly cutting off the oldest grain. Because this truncation occurs mid-waveform rather than at a zero-crossing or envelope boundary, it produces a sharp transient. Disabling voice stealing (@steal 0) eliminated the clicks but caused new grain triggers to be silently ignored, producing audible dropouts at high density.

Neither the default behaviour nor the obvious alternative was acceptable. The solution required re-engineering the voice architecture entirely. A secondary micro-fade logic using line~ was implemented to intercept the control signal when the active voice count approached maximum capacity. Rather than allowing the system to steal a voice abruptly, the oldest active grains are forced into an accelerated 5ms fade-out envelope before the voice is reallocated. At 5ms the fade is imperceptible to the listener but long enough to bring the waveform smoothly to zero, masking the transient entirely.

Dynamic envelope management on voice reallocation. 5ms forced fade-out on the oldest grain before voice stealing occurs. The engine operates at maximum density without DSP artifacts.

Bridging the two engines via the audio-reactive feedback loop caused the system to either collapse into silence or exponentially scale into maximum density within seconds of starting. This was the most fundamental problem in the project because it threatened the entire architectural premise.

Linear 1:1 mapping in a positive feedback loop is inherently unstable. High granular density raises the master RMS; a linearly mapped RMS raises melodic note density; higher note density raises the RMS further. The system enters a runaway state. The inverse is equally destructive: a quiet moment lowers the RMS, which lowers density, which lowers the RMS further, driving the system to silence. There is no stable equilibrium.

Two specific interventions were required to solve this.

### Data dampening

slide and rampsmooth~ objects were introduced into the control signal path, acting as low-pass filters for the data rather than the audio. This forces the system's reaction time to behave like a physical mass: a sudden spike in audio energy takes several seconds to slowly push the sequencer parameters upward, and vice versa. The system can still respond to sustained changes in its audio output, but transient spikes in either direction no longer destabilise the whole system.

### Exponential bounding

Linear scale objects were replaced with logarithmic mapping curves. As the system gets louder, it requires exponentially more audio energy to push the parameters higher. This creates a natural mathematical ceiling; the system can approach maximum density but the feedback loop loses leverage as it gets closer to it, preventing the runaway state entirely.

Non-linear scaling combined with data dampening. The feedback loop now has stable equilibrium points at multiple density levels. The system settles into evolving states rather than collapsing or exploding.

## The Visual Engine

The visual patch translates the same RMS and spectral analysis data controlling the audio engines into real-time visual feedback via Jitter. Audio-reactive parameters (mesh scale, colour saturation, noise distortion amount) are driven directly from the control-rate values already present in the system; no separate analysis is required.

### CPU vs GPU processing

Early iterations used standard jit.matrix objects for visual processing, which runs on the CPU. Because Max/MSP's audio thread demands CPU priority, complex visual processing at this stage caused audio dropouts; the visual and audio pipelines were competing for the same resource.

The fix was migrating the entire visual pipeline to the GPU using jit.gl.* objects (jit.gl.slab for shader processing, jit.gl.pix for pixel-level operations). All pixel math runs on the graphics card, freeing the CPU to handle the audio thread without interruption. The audio-visual synchronisation remains tight because both systems read from the same control-rate data; the only change is where the visual computation happens.

## Interface Design

A system this complex is useless if the interface makes it impenetrable in performance. The UI was designed to emulate physical hardware; dark, high-contrast, built for live use rather than deep editing sessions.

### Macro controls

Rather than exposing every DSP variable directly, the UI uses carefully scaled macro controls that adjust multiple parameters simultaneously under the hood. The Mood system, for example, maps a single dial to grain size, pitch randomisation range, and melodic clock speed in tandem. The performer shapes the character of the system without needing to understand the individual parameters driving it.

### Constrained randomisation

A "Surprise Me" function recalculates the system's baseline parameters within predefined safe ranges. Purely random mapping to all variables produces silence or feedback; the randomisation logic uses random combined with scale nodes to generate new values only within ranges that are guaranteed to produce stable, musical output. It gives the system a new starting point instantly without destabilising the audio engine.

### State interpolation

Max's pattrstorage system is used to save and recall UI states. Interpolating between two presets over a set duration (5 seconds by default) forces the poly~ and sequencing logic to transition smoothly from a dense texture to a sparse melody rather than snapping between them, creating structural shifts that feel composed rather than switched.

## Outcomes

The three engineering problems SONIN encountered (clock drift, voice stealing artifacts, feedback instability) are not unique to this project. They are fundamental challenges in any system that uses real-time feedback to drive generative behaviour. The solutions (sample-and-hold quantisation, dynamic micro-fade envelope management, non-linear logarithmic bounding) are transferable patterns that apply directly to any autonomous generative system.

The more interesting outcome is what the project confirmed about the design principle itself. Generative audio does not have to mean unpredictable noise. Applying strict mathematical constraints, careful feedback dampening, and proper DSP problem-solving produces a system that composes with genuine musical coherence; one that sounds like it is making decisions rather than producing random output.
