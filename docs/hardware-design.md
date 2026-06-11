---
tags: [effie, hardware, doll-build, stage-rig]
status: planning
updated: 2026-06-11
---

# Effie Hardware Design — Doll Build & Stage Rig

Working notes from design discussion 2026-06-11. Software is feature-complete
(all PRD stories done); this covers the physical build. **Guiding principle:
hardwired everything — zero RF on the control path.** Theaters are hostile RF
environments (wireless mic banks, house WiFi, lighting consoles, phones);
a copper pair is deterministic and inspectable in seconds at setup.

---

## 1. System overview

```
 PERFORMER                          DOLL TABLE                      VENUE
┌─────────────────┐      ┌────────────────────────────┐
│ Handheld mic    │      │  Junction box (doll base)  │
│  + FSR button   │ star │   ├─ audio pair ──────────────► XLR out ──► house PA
│  (passive, no   │═quad═│   ├─ iso-split audio ─► USB │
│   electronics)  │cable │   │   audio interface       │
└─────────────────┘      │   └─ button pair ─► Pico ───┤
                         │            ▲                │
┌─────────────────┐      │            │ GPIO           │
│ Doll pull-string│──────│── IR sensor┘                │
│   mechanism     │      │   Pico = USB HID keyboard   │
└─────────────────┘      │        ──► mini PC (Effie)  │
                         └────────────────────────────┘
```

Both physical inputs reach Effie as **keyboard events** (Pico acting as a USB
HID keyboard). The software never changes — it already listens for F12/F11.

| Input | Key event | Effie behavior |
| --- | --- | --- |
| Pull string returns home | F12 tap (100 ms) | Plays front of audio queue |
| Mic button squeeze-hold | F12 down/up (real timing) | Record → decode → answer (queued) |
| Mic button double-squeeze-hold | F12 pattern | Record, bypass decoder |
| (optional second sensor) | F11 long press | Phase advance |

---

## 2. Doll: pull-string trigger

**Mechanism:** salvage the real pull-string reel from the donor doll.

**Sensor:** Adafruit STEMMA Reflective Photo Interrupt — TCRT1000
([product 5913](https://www.adafruit.com/product/5913), $4.95).
Active IR emitter + phototransistor; immune to stage lighting, especially
inside the closed doll body (dark cavity, only its own IR present).

Key facts learned:
- **Sensing range is millimeters (~1–4 mm)** — placement is precise. Aim at
  the cord-stop or reel surface at the string's home position.
- **Use DIFFUSE reflector, not shiny paint.** Specular (mirror/chrome) surfaces
  bounce IR at one exact angle and miss the detector when tilted. Best→worst:
  **retroreflective tape** (bike/safety tape — returns light at any angle) >
  flat white paint > glossy/chrome paint.
- Onboard pot adjusts emitter brightness (trigger distance tuning); onboard
  detection LED gives visual feedback while calibrating. LEDs disableable.
- Output: high when idle, drops toward 0 V on detection. 3–5 V supply,
  JST PH STEMMA connector.
- Read with **Pico ADC** (not bare digital) → hysteresis in software
  (trigger below X, re-arm above Y) + 50 ms "settled home" debounce.

**Trigger on return-to-home, not on pull** — gives the retraction a beat of
mechanical silence before she speaks, like the vintage toys.

**Build order:** desk rig first. Sensor + Pico + salvaged reel clamped to the
desk, retroreflective tape on the cord-stop, watch the detection LED through a
few hundred pulls *before* anything goes into the doll body. The vintage
mechanism's mechanical endurance is the real unknown, not the electronics.

---

## 3. Performer: mic-mounted button (hardwired)

**Sensor choice: FSR (force-sensitive resistor) over capacitive.**
Capacitive fires on *contact* and the hand is always in contact with a
handheld mic — false trigger machine. An FSR under a specific thumb spot
with a squeeze threshold (read via ADC) distinguishes deliberate pressure
from grip pressure. Invisible, deliberate, un-triggerable by accident.
One FSR serves the whole gesture grammar (hold / double-squeeze-hold);
an optional second spot = F11 phase change.

**The one-cable trick: star-quad.**
Standard mic cable = 3 conductors. **Star-quad (Canare L-4E6S, ~$1/m)** =
4 conductors + shield, visually identical to any stage mic cable:

| Conductor | Job |
| --- | --- |
| Pair 1 | Balanced mic audio |
| Pair 2 | FSR/button circuit |
| Shield | Audio ground |

- Mic end is **completely passive** — no battery, no radio, no electronics.
  All smarts live at the junction box end.
- Connector: **5-pin XLR** (DMX/AES style — indistinguishable from a mic plug
  at audience distance) if detachable; or hardwire into a grip sleeve —
  one less failure point.
- Phantom power (if condenser mic): passes straight through on the audio
  pair; button pair is electrically separate — 48 V never meets the Pico.
- Noise: FSR is resistive (no switching edges), star-quad geometry rejects
  coupling, button circuit is 3.3 V at microamps. If a tick is ever audible,
  a small cap across the FSR kills it.
- A wired handheld is **period-appropriate stagecraft** for an antique-doll act.

**Cost:** tethered to the doll table by cable length; can't hand the mic into
the audience. Acceptable if blocking stays near the doll.

---

## 4. Audio path (voice → Effie AND house PA)

Junction box at the doll table:

1. Audio pair → **female XLR jack** = the venue handoff. Sound tech plugs in,
   gets a normal mic feed, asks zero questions.
2. **Transformer-isolated split** (ART DTI or Radial passive DI, ~$50–70)
   taps the same audio for Effie without ground-loop hum.
3. Tap → **cheap USB audio interface** (~$30, e.g. Behringer UCA202/UM2) →
   mini PC. Set as default input; Effie's STT (ElevenLabs Scribe) is happy
   with modest audio quality — broadcast quality only matters for the house.

---

## 5. Parts list

| Part | Purpose | Est. cost | Have? |
| --- | --- | --- | --- |
| Donor doll w/ real pull-string reel | Mechanism + body | — | ✅ |
| Adafruit TCRT1000 STEMMA (5913) | String home sensor | $5 | ✅? |
| Retroreflective tape | String/reel marker | $5 | |
| Raspberry Pi Pico (any variant) | Sensors → USB HID keyboard | $5 | ✅ (PiPicoDevFolder exists) |
| FSR (e.g. Interlink FSR-402) | Mic squeeze button | $7 | ? |
| Canare L-4E6S star-quad cable | Mic audio + button, one cable | ~$1/m | |
| Neutrik 5-pin XLR pair (optional) | Detachable mic cable | $15 | |
| ART DTI / Radial passive DI | Isolated audio split | $50–70 | |
| USB audio interface | Effie's ears | $30 | |
| Small project box | Junction box in doll base/table | $10 | |
| Mini PC | Runs Effie | — | ✅ |

---

## 6. Open questions

- [ ] Which Pico variant is on hand? (any works for USB HID; W not needed — no wireless)
- [ ] Can the handheld mic body be opened, or build a grip sleeve?
- [ ] Roaming distance from doll table during the act? (sets cable length)
- [ ] Wired mic model to use? Dynamic vs condenser (phantom passthrough)?
- [ ] Mini PC model / does Effie's table have power onstage?
- [ ] Does the doll table hide the junction box + mini PC, or does the doll body?
- [ ] Mechanical endurance of the vintage reel — needs the desk-rig test.
- [ ] Optional: discreet "answer is ready" cue for the performer (vibration
  motor / LED sightline / earpiece click) — queued mode gives no audible cue.

## 7. Deferred / fallback options

- **BLE HID (ESP32 or Pico W in mic)** — rejected for v1 due to theater RF
  congestion, but fully designed: mic-side battery + BLE keyboard, same F12
  semantics. Revisit only if the tether proves unworkable.
- Bluetooth camera-shutter remotes — rejected: unreliable press/release
  (hold) semantics, wrong keycodes.
- Shiny-paint + passive photosensor — superseded by active IR (TCRT1000)
  + diffuse reflector.
- Mid-cable DC signaling over a standard 3-pin XLR (phantom-style) —
  clever but interacts with house phantom; star-quad spare pair is cleaner.

## 8. Next actions

1. Desk rig: salvaged reel + TCRT1000 + Pico → CircuitPython sketch
   (ADC read → hysteresis → debounce → F12 HID tap). ~30 lines; ask Claude.
2. Endurance-test the reel (few hundred pulls, watch detection LED).
3. Choose/buy: FSR, star-quad cable, DI, USB interface.
4. Mic teardown or sleeve decision → mount FSR.
5. Junction box build + full chain rehearsal with Effie in queued mode.
