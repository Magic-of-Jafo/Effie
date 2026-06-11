# Mini-PC Deployment Prompt

Paste the block below into Claude Code on the target machine (e.g. the
performance mini PC) to set up and verify Effie from scratch. Written
2026-06-11, when the app was feature-complete and verified on the dev machine.

---

# Deploy Effie (CorindaGPT) on this mini PC

## What this project is
This machine will run **Effie** — a headless Python asyncio app that acts as a
hidden AI accomplice in a live magic act (a pull-string doll). It listens to the
magician's mic on a hotkey, transcribes via ElevenLabs Scribe, decodes secret
code phrases from a CSV (Hadley system), asks an LLM (gpt-5.4-nano), and speaks
replies in a custom ElevenLabs voice. It was made feature-complete and fully
verified on the dev machine on 2026-06-11; your job is **deployment and
environment verification only — do not change application code or config
without asking me first.**

Read `docs/architecture.md`, `docs/prd.md`, and `docs/hardware-design.md`
after cloning for full context. Story history is in `docs/stories/`.

## Step 1 — Clone
git clone https://github.com/Magic-of-Jafo/Effie.git into an appropriate
projects directory and work from the repo root.

## Step 2 — Python environment
- Requires **Python 3.11** (3.13 will NOT work: the optional `simpleaudio`
  dependency has no wheels for it, and 3.11 is what's verified). If 3.11 is
  not installed, install it (winget or python.org) before proceeding.
- Create the venv at the **repo root**: `py -3.11 -m venv .venv`
- Install: everything in `corindagpt/requirements.txt` EXCEPT `simpleaudio`
  (skip it — it needs a C compiler and is only a deep fallback; playback uses
  sounddevice), PLUS `sounddevice` and `numpy`:
  filter requirements.txt to exclude the simpleaudio line, then
  `.venv\Scripts\python -m pip install -r <filtered> sounddevice numpy`

## Step 3 — API keys
Create `corindagpt\.env` (it is gitignored) containing:
  OPENAI_API_KEY=...
  ELEVENLABS_API_KEY=...
Ask me for the two values when you get there — do not guess or reuse anything
found on disk.

## Step 4 — Verify, in this order
1. **Tests**: run pytest from the `corindagpt\` directory (NOT repo root —
   imports resolve via its conftest): `..\\.venv\Scripts\python -m pytest tests -q`
   Expect 33 passing.
2. **Audio devices**: list devices with sounddevice and show me the defaults
   (`python -c "import sounddevice as sd; print(sd.query_devices())"`).
   This machine's mic/speaker setup is unverified — I'll confirm which
   devices are correct before live tests.
3. **API smoke test**: from `corindagpt\`, run a short script that calls
   `src.services.gpt.generate_response` ("Reply with exactly: pipeline check OK")
   and `src.services.tts.synthesize` on a short phrase, then plays it via
   `src.services.tts.play`. I should hear Effie's voice from the speakers.
4. **Round trip**: feed the synthesized audio into
   `src.services.voice_to_text.TranscriptionService.transcribe` and confirm
   the text comes back.
5. **Decoder check** (no API cost): from `corindagpt\` run
   `..\\.venv\Scripts\python -m scripts.decode_inspect "Cool, could you tell me what card this is?"`
   — expect Playing Card Value: King of Hearts in the structured output.
6. **Latency benchmark** (optional but useful on new hardware/network):
   `..\\.venv\Scripts\python -m scripts.latency_bench`

## Step 5 — Live run
Launch with debug logging so I can watch:
  cd <repo root>
  $env:CORINDA_DEBUG = "1"
  .venv\Scripts\python -m corindagpt
Wait for "OpenAI connection warmed up" / "ElevenLabs connection warmed up",
then I'll test the gestures myself:
- Tap F12 → plays next pre-recorded doll phrase (11 in assets/phrases/)
- Hold F12 + speak coded phrase → answer is synthesized and QUEUED (config is
  in doll mode: `audio_queue.response_playback: "queued"`), then tap F12 to
  hear it
- Double-tap+hold F12 + speak → bypasses the decoder, answers literally
- Hold F11 3s → phase advance
Note: in queued mode there is NO immediate audio after a question — that is
correct behavior, watch the log for "AudioQueue: prioritized".

## Known gotchas (already handled in code — just don't "fix" them)
- gpt-5.4-nano needs max_completion_tokens/reasoning_effort instead of
  max_tokens/temperature, and reasoning_effort must be dropped when tools are
  attached. `_completion_params` in src/services/gpt.py handles this.
- pytest only works from the corindagpt/ directory.
- All logging is suppressed unless CORINDA_DEBUG=1 (stealth is intentional).
- mpv is NOT required — streaming playback goes through sounddevice.
- The `.history/` folder and `last_tts.wav` are gitignored debris if they appear.

## Step 6 — When everything passes
Propose (but don't implement until I approve) a headless autostart setup for
performances: launch on boot/login without a console window, CORINDA_DEBUG off.
