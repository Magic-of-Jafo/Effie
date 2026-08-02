---
tags: [effie, voice-control, prd, draft]
status: draft — design discussion, not scheduled
updated: 2026-08-02
---

# Mini-PRD: Voice-Controlled Operation

Replaces the physical control path (mic push-button, Pico HID, custom cable
— ~$500 of hardware) with spoken trigger phrases. The **audio path is
unchanged**: performer's wireless mic → receiver → USB interface → mini PC.
The doll's pull-string trigger is unchanged.

Nothing here is scheduled. This is the shared picture to argue with before
any code is written.

## 1. Goals

- Hands-free triggering: no button, no wearable, no cable to the performer.
- Never fire when not intended. A false positive on stage (the doll speaking
  over the performer, or revealing at the wrong moment) is unrecoverable; a
  missed trigger costs one repeated line.
- Graceful, in-character recovery from every failure mode.
- Lower latency than the current button flow (~1.8 s release-to-queued).
- All phrases and behavior operator-editable in data files — the system may
  be sold to other performers.

## 2. Foundation

OpenAI **`gpt-live-transcribe`** (GPT-Realtime-Whisper, released 2026-05-07):
continuous WebSocket, 24 kHz PCM in, incremental transcript deltas out.
~$0.017/min ≈ $0.50 for a 30-minute show. Transcript exists as the performer
speaks, so processing can begin the moment a trigger closes.

## 3. Locked design decisions

### 3.1 Strict AND bracket

```
WAKE PHRASE … payload sentence(s) … RELEASE PHRASE
```

Both markers required, in that order. Everything between them is the buffer.

- **Omitting either marker means "do not fire."** No degraded fallback, no
  look-back guessing, no pause timeout that fires anyway. Omission is
  treated as intent, not error.
- A stale wake phrase silently disarms after a timeout (configurable, ~20–30 s)
  so a forgotten arm cannot be closed by a stray release phrase later.
- Markers are stripped from the buffer before decoding.
- **Release phrases must end a sentence.** Mid-sentence occurrences do not
  close the bracket.
- **Whole-phrase matching only.** "Effie" alone must never arm — the doll can
  be addressed by name freely in patter. Only the complete wake phrases
  ("Effie focus", "Effie listen", "Effie look") match.
- Multi-sentence codes (clock time, calendar date, future 3-part codes) need
  no special machinery — they are simply multiple sentences inside one
  bracket. No prefix tree, no combo window, no pause timer.

### 3.1.1 Multiple answers per window, with accumulating context

Every sentence inside the bracket produces its own answer MP3, queued in
spoken order (existing FIFO priority). Each successive answer can see all
decoded data accumulated so far in that window.

**Same-row follow-ups** (second sentence has no code phrase):

> "Listen carefully. Guess what the man is holding. Be sure to include the
> color. Got it?"
> → pull 1: "The man is holding a wallet." → pull 2: "The wallet is green."

**Progressive reveal** (second sentence carries its own code phrase —
replaces any need for a time/date consolidation path):

> "Guess what time the watch stopped." → "The hour is four o'clock."
> "All right, name the exact minute." → "I can see the time is exactly
> four thirty-three."

> "Guess which month this happened." → "I see the month of March."
> "Now guess the day." → "Yes, March the thirteenth is the date."

One universal rule — one sentence in, one MP3 out, context accumulates —
covers same-row follow-ups, clock time, calendar date, and any future
multi-part code. No special-case workflow.

*Tradeoff:* a single unspoiled time/date climax is not possible this way,
since both require two coded sentences to encode. Consolidation into one
MP3 remains an option if a routine needs it.

### 3.1.2 Code phrase position

A code phrase must **begin the sentence** — longest match starting at word
one. A code phrase appearing later in a sentence is ignored.

> "Please now try to focus." → no match (nothing starts with PLEASE;
> `NOW TRY` at words 2–3 is invisible)

Stricter than the current implementation (leftmost match anywhere) and
strictly safer: it retires accidents like `TELL` matching inside "could you
tell me". All existing decoder tests pass under this rule.

### 3.2 Error phrases as a covert diagnostic channel

Pre-recorded MP3s in Effie's voice that the audience hears as character and
the performer hears as system status. Selection is **state-driven**, not a
blind rotation — the system knows why nothing landed.

| System state | Performer learns | Draft phrase |
| --- | --- | --- |
| Network/API unreachable | Show cannot continue as planned; switch to non-AI material | "I'm not feeling the connection." |
| Request in flight **or** nothing triggered (no bracket detected) | Restart the request: wake phrase and ask again | "Things aren't clear. Let me concentrate." |
| Processed but produced nothing (decode miss, empty/failed response) | Re-ask with a cleaner code phrase | "The image is… confusing." |

Three states, not four: "in flight" and "never triggered" deliberately share
a phrase because the performer's recovery is identical (start over). Known
consequence — if the request really was in flight, restarting stacks a
duplicate answer; the flush command (§3.4) clears it.

Requirements:
- Multiple variants per state so repeats don't sound identical.
- Written as **effort, not apology** — "let me concentrate" plays as
  character; "I'm sorry, I don't know" plays as a bug.
- Covers the early-pull case: assistant pulls before the answer lands →
  error phrase plays → answer arrives → next pull delivers it. The failure
  becomes a dramatic beat (the doll straining, then succeeding).

### 3.3 Three-tier audio queue

1. **Answers** — LLM responses, FIFO among themselves (existing behavior).
2. **Error phrases** — always-available default, state-selected (§3.2).
3. **Ambient doll phrases** — the existing 11 character lines, for
   non-interrogation stretches of the act.

A pull takes the highest available tier.

### 3.4 Flush command

Saying **"sorry"** and **"clear"** in close proximity (e.g. "Sorry, let's
clear our minds…") is a full reset:

- removes all queued answer MP3s, leaving the default tiers, **and**
- clears the carried row.

Extends the existing "sorry" convention, which already voids everything
before it in a decode. After a flush, the next question must carry its own
code phrase — nothing is inherited.

### 3.5 Input ducking during playback

Whenever Effie speaks — an answer, an error phrase, or an ambient line —
the system ignores incoming speech for the duration of that audio. This is
the defense against self-triggering (§6): her own error phrase contains
"concentrate", a release phrase, and answers can begin with code words
("**Can** you see it?").

- **Anchored to playback, not to the pull.** The string pull is
  instantaneous; the audio has duration, and that duration is known exactly
  before playback begins. Blind spot = playback start → playback end +
  short tail (~200 ms) for room reverb.
- **Duck at the matcher, not the microphone.** Audio keeps streaming and
  transcribing; the transcript from the blind-spot window is simply not
  eligible to trigger anything. This costs nothing extra and means rehearsal
  logs (§3.6) can show exactly what was heard-but-ignored — the only way to
  find out whether the blind spot is eating real content.
- **Known cost:** anything the performer says while she is speaking is lost
  as a trigger. Her lines are short (persona caps ~10 words), so the window
  is 2–4 seconds, but a wake phrase spoken over her tail will not register.
  Rehearsal logging will show whether this happens in practice.

### 3.6 Rehearsal logging mode

A mode that records a reviewable timeline of a full run-through:

- every transcript line, with timestamps
- every wake / release / flush phrase matched (and near-misses worth seeing)
- what was captured in each window, what decoded, which row was active
- what would have fired vs. what did
- blind-spot windows and any speech heard inside them
- error-phrase selections and why

Purpose: find phrases that misfire in real patter, phrases that transcribe
unreliably, content lost to blind spots, and code words appearing
accidentally at sentence-start. Reviewed between rehearsals; the phrase
pools and rules get tuned from evidence rather than guesswork.

## 4. Phrase pools (starting set)

Hard rule: **no code words** in any trigger phrase, or the matcher and the
decoder can fight over the same words. Excluded: *try, can, think, guess,
are, give, say, name, see, tell, what, do, could, now, so, well, all right,
okay, cool, please, then*.

All phrases below verified clean against `effie_code.csv`.

**Testing seeds only** — the performer will expand both pools so rotation
never reveals the method to the audience.

**Wake**
- Listen carefully…
- Effie focus…
- Effie listen…
- Effie look…

**Release**
- …when you're ready.
- …if you're ready.
- …got it?
- …concentrate.

The name-directed wake phrases are deliberate: addressing the doll is
natural stagecraft in a telepathy act. The name alone never arms (§3.1).

Both pools live in an external editable file. Selection criteria for future
additions: natural in that slot, **not** already a habitual verbal tic
(a phrase you say without meaning to is a bad marker), transcribes
reliably, and distinct from the other pool.

## 5. Preserved behaviors

- **Active-row carry**: an armed, code-free question answers about the
  previously established item. This is a deliberate misdirection tool — the
  audience hears no code at all.
  - **Lifetime**: the active row persists until a new code phrase is
    detected **inside an activation window**. Code words spoken outside a
    bracket never change it.

  > "Effie look. What color is the wallet? When you're ready."
  > → no code phrase → still the previously established row → "green"
- **"Sorry"** voids preceding content in a capture.
- **Button/keyboard input stays live** in every mode as manual override.
- **Queued (doll) playback mode** unchanged: answers wait for the string.

## 6. Risks

| Risk | Notes / mitigation |
| --- | --- |
| **Self-trigger via speaker bleed** — "concentrate" and "clear" appear in Effie's own error phrases; answers can begin with code words | **Resolved by input ducking (§3.5)**: playback-anchored blind spot at the matcher. Residual: performer speech during her lines is lost as a trigger |
| Accidental trigger phrases in patter | Mitigated by strict AND (both markers, in order, around codeable content) |
| Network loss mid-show | Dedicated error phrase (§3.2) + button fallback + local failure detection |
| STT transcription drift on trigger phrases | Normalization map (as with ALRIGHT→ALL RIGHT); verify each phrase against real audio |
| Wake pool leaning on "Effie" (3 of 4) | Name repetition was previously rejected as audience-detectable — see open questions |

## 7. MVP scope

The first build exists so the performer can **rehearse the code act
hands-free** and get fluent before an audience ever sees it. Everything not
serving that is deferred.

**In:**
- Streaming transcription (§2)
- Wake/release bracket, strict AND (§3.1), sentence-initial codes (§3.1.2)
- Capture → existing decode → LLM → TTS → queue pipeline
- Multiple answers per window with accumulating context (§3.1.1)
- Error phrases, three states (§3.2)
- Input ducking (§3.5) — without it, self-triggering breaks everything
- Rehearsal logging (§3.6) — the point of the MVP is learning from it
- Flush command (§3.4) — the safety valve
- Button/keyboard override stays live throughout

**Deferred:**
- Phase changes by voice (see §8)
- Performer "answer landed" cue hardware (see open questions)
- Any dashboard / operator UI

## 8. Deferred: voice phase changes

Later build, once the act has grown into multiple phases. Sketched design:
**activation phrase + a dedicated phase code phrase → switch immediately,
no release phrase required.** The absence of a release marker is what
distinguishes a phase switch from a question. F11 long-press remains the
manual fallback.

## 9. Open questions

- [ ] **Performer status cue.** Currently flying blind — the error phrases
      are the only feedback, and they are *reactive* (you learn only after
      pulling the string). Wanted: something proactive and discreet
      (pocket thumper, small light) confirming the system is healthy and an
      answer has landed. Unsolved; constrained by the no-wireless,
      no-wearable-cable preferences. One option that fits: a small LED at
      the doll's table, angled to be visible only from the performer's
      position — the mini PC is already there, so it needs no new run to
      the performer.

### Resolved

- Release phrases must end a sentence. ✅
- Name alone never arms; whole wake phrases only. ✅
- Carried row persists until a new code is detected inside a window. ✅
- "Nothing triggered" and "in flight" share one error phrase. ✅
- Name-directed wake phrases are intentional. ✅
- Code phrases must begin the sentence (§3.1.2). ✅
- Time/date use progressive reveal via accumulating context, not a
  consolidation path (§3.1.1). ✅
- Leading fillers: **strict position-zero, no skip-list.** This matches how
  the system has always behaved. Revisit only if rehearsal shows it costing
  real triggers. ✅
- Flush clears queued MP3s **and** the carried row (§3.4). ✅
- Input ducking during playback (§3.5) — applies to **all** doll audio:
  answers, error phrases, ambient lines, any sound she makes. ✅
- Rehearsal logging mode (§3.6). ✅
- `TRY` at sentence-start stays live (row 101, Joker). Handled by performer
  discipline: never open a sentence with "try" unless row 100/Joker is
  intended. ✅
- Voice phase changes deferred to a later build (§8). ✅

## 10. Non-goals

- Speaker diarization / distinguishing performer from audience voices.
- Wake-word-free operation (rejected: false positives).
- Degraded single-marker triggering (rejected: §3.1).
- Replacing the doll's pull-string mechanism.

## 11. Provisional story breakdown

1. **Streaming transcription service** — WebSocket client, rolling
   transcript, sentence assembly, reconnect, network-loss detection.
   Independently useful: speeds up button mode too.
2. **Trigger phrase matcher** — external phrase file, wake/release/flush
   classes, normalization, marker stripping.
3. **Capture engine** — strict AND bracket, arm timeout, buffer handoff to
   the existing decode → LLM → TTS → queue pipeline.
4. **Error phrase system** — state-driven selection, third queue tier,
   audio generation script for the variants.
5. **Self-trigger suppression** — gate transcription during doll playback.
6. **Validation tooling** — rehearsal logging and review before stage use.
