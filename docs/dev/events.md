# Events (design note)

Date: 2026-11-23. Status: option B implemented, with both `@event` and the `on_` prefix required, and the proposed answers to 2 to 4 (Pd and SuperCollider both done). Spec 3.3, 6 and 8 hold the rules.

Question: how does a host tell a kernel "note 60 on, velocity 0.8"?

## Problems

Found by `tests/fixtures/poly.py` (`TODO.md`):

1. **No retrigger.** A gate is a parameter. Re-sending a held gate's value changes nothing.
2. **Collapsed events.** A note-off and a note-on in one block leave only the last value.
3. **Allocation lives in the host.** The kernel cannot take a note number; the host picks a voice and sets `freqs_k` and `gates_k`.

Timing is not a problem. Output does not depend on block boundaries (spec 9.2). A host can therefore split `process` at an event's frame and get sample-accurate timing with no ABI change.

## Option A: a "changed" query

`self.changed("gates", i)` in `control` is true if the host set the slot since the last block, even to the same value.

- Fixes 1.
- Fixes 2 only if the host calls `process(n=0)` between the two sets. That is a host convention, not a contract.
- Does not fix 3.
- ABI: none. The setter sets a flag; `process` clears it.

## Option B: event methods

A method marked `@event` becomes a function hosts call between `process` calls:

```python
class Poly(Kernel):
    voices: Array[Voice, 8]
    next: i32 = 0

    @event
    def note_on(self, note: i32, velocity: f32) -> None:
        v = self.next
        self.voices[v].start(note, velocity)
        self.next = (v + 1) % 8

    @event
    def note_off(self, note: i32) -> None:
        for i in range(8):
            if self.voices[i].note == note:
                self.voices[i].release()
```

- Fixes 1, 2 and 3. Each call runs, in order, and the kernel allocates voices in checked Python.
- ABI additions, following the parameter pattern:
  - a named function per event: `void Poly_note_on(Poly *self, int32_t note, float velocity)`;
  - an indexed dispatcher for shims: `void Poly_event(Poly *self, int index, const float *args)`;
  - `info.events`: name and argument count per event.
- Shims:
  - Pd: one method per event on the left inlet (`note_on 60 0.8`), as `reset` is now.
  - SuperCollider: `DefineUnitCmd` per event (`SC_InterfaceTable.h:74`), sent as `/u_cmd node unit "note_on" 60 0.8`.
- Tests: `Case.changes` already holds frame-stamped `set` and `reset` entries. An event entry fits the same list, so the differential, block-pattern, Pd and SC tests cover events with no new harness.

## Recommendation

B. A fixes only the first problem. B fixes all three, and its ABI addition copies an existing pattern: a named setter beside an indexed one. Parameters stay as they are, for continuous controls.

## Decisions needed

1. **Marker.** `@event` decorator (explicit; proposed) or a name prefix such as `on_` (no import).
2. **Argument types.** `f32`, `f64`, `i32` and `bool` (proposed). The dispatcher gets `sample` values. Converting to `i32` needs defined NaN and out-of-range behaviour, since C's float-to-int conversion is undefined there. Proposed: truncate towards zero, saturate, and map NaN to 0.
3. **What an event may read.** Proposed: anything, including smoothed parameters. An event runs at a defined frame, not at a block start, so its result does not depend on block size. That holds only if hosts deliver events at a frame boundary, which both shims do (inference: Pd runs messages between DSP ticks, scsynth runs unit commands between blocks; not yet checked in source).
4. **SuperCollider idiom.** SC users usually get polyphony from one synth per voice, with the language allocating. Event methods still help there (retrigger), but allocation inside one UGen is an unusual pattern for SC users. Ship it for both hosts, or Pd first?

## Not proposed

- **MIDI in the ABI.** Fixed `note_on`/`note_off` signatures would fit fewer kernels. A shim can map MIDI to event methods by name; gen-dsp maps note-on and note-off to `note(pitch, velocity)` and `off(pitch)`.
- **Event queues inside `process`.** Sample-accurate timing already comes from splitting blocks.
