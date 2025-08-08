# CorindaGPT Architecture Document

| Date | Version | Description | Author |
| :--- | :--- | :--- | :--- |
| 2025-08-07 | 1.0 | Initial architecture draft creation. | Winston (Architect) |

## 1\. Introduction

This document outlines the overall project architecture for CorindaGPT, including backend systems, shared services, and non-UI specific concerns. Its primary goal is to serve as the guiding architectural blueprint for AI-driven development, ensuring consistency and adherence to chosen patterns and technologies.

### Starter Template or Existing Project

Not applicable. This is a greenfield project being built from scratch to custom specifications.

## 2\. High Level Architecture

### Technical Summary

The CorindaGPT system will be a headless, monolithic Python application architected around a central `asyncio` event loop to achieve the critical goal of low-latency, real-time processing. Its key components include a hardware-agnostic input handler, an in-memory state machine for managing performance phases, a dynamic prompt loader using `langchain-core`, and an audio processing pipeline. This single-process, stateful design directly supports the PRD's primary goal of serving as a reliable and high-speed secret assistant for a live magic performance.

### High Level Overview

As established in the PRD's Technical Assumptions, the architecture will adhere to the following principles:

  * **Architectural Style**: **Monolith**. The system is a single, cohesive application. Its components, while modular, will run within a single process.
  * **Repository Structure**: **Polyrepo**. The entire codebase will be housed in a single Git repository for simplicity and ease of management.
  * **Primary Interaction Flow**: The application's core logic follows a clear sequence: A discreet physical input is detected and interpreted by the **Input Handler**, which triggers a transition in the **State Machine**. This initiates the main `asyncio` pipeline: **Audio Capture** -\> **Transcription** -\> **Decoding** (if applicable) -\> **Dynamic Prompt Generation** -\> **LLM Interaction** -\> **TTS Generation** -\> **Audio Playback**.

### High Level Project Diagram

```mermaid
graph TD
    subgraph CorindaGPT Application (Mini PC)
        A[Input Handler] --> B{State Machine};
        B -- Manages --> C[Main asyncio Loop];
        C -- Reads --> D[config.yaml];
        C -- Loads --> E[Prompt Files];
        C -- Uses --> F[Decoder Modules];
        F -- Loads --> G[Decoder Data Files];
        C -- Uses --> H[langchain-core];
        
        C -- Calls --> I[OpenAI API];
        C -- Calls --> J[ElevenLabs API];
    end

    Magician -- Discreet Input --> A;
    I -- LLM Response --> C;
    J -- TTS Audio --> C;
    C -- Audio Output --> Speaker;

    subgraph External Services
        I;
        J;
    end
```

### Architectural and Design Patterns

  * **Overall Architecture: Stateful `asyncio` Process**
      * We will implement the application as a single, stateful, long-running `asyncio` process.
      * **Rationale**: This is the simplest and most effective pattern for managing conversational state, performance phases, and low-latency hardware interactions. It avoids the potential for "cold starts" that could be introduced by a serverless architecture, which is critical for a live performance.
  * **Code Organization: Modular Monolith**
      * The codebase will be organized into distinct, single-responsibility Python modules (e.g., `voice_to_text.py`, `state_machine.py`, `gpt.py`) within a single application.
      * **Rationale**: This provides a clear separation of concerns, making the code easier to develop, test, and maintain, while avoiding the unnecessary complexity of a microservices architecture.
  * **External Communication: Direct Asynchronous API Calls**
      * All communication with external services (OpenAI, ElevenLabs) will be done via direct, non-blocking API calls using `asyncio` and `httpx`.
      * **Rationale**: This pattern provides the lowest possible overhead and latency, directly addressing the core non-functional requirement (NFR1) for speed.
  * **State Management: In-Memory State Machine**
      * The application's status (`IDLE`, `LISTENING`, etc.) and the current performance phase will be managed by a simple, in-memory state machine.
      * **Rationale**: This provides the fastest possible state access, which is perfectly suited for a single-process application where state does not need to be persisted between runs.

## 3\. Tech Stack

### Deployment Host & External Services

  * **Deployment Host:** Standalone Mini PC.
  * **External Services:** The application relies on external APIs for its core AI functionality:
      * **OpenAI API**: For Large Language Model processing.
      * **ElevenLabs API**: For high-quality Text-to-Speech generation.
  * **Cloud Provider / Regions:** Not applicable, as the core application is self-hosted.

### Technology Stack Table

| Category | Technology | Version | Purpose | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Language** | Python | 3.12.x | Primary development language | Modern, stable, with excellent support for `asyncio` and AI libraries. |
| **Package Mgmt** | pip | latest | Manages Python dependencies | Standard for Python; `requirements.txt` will be used for version pinning. |
| **Async Framework**| asyncio | stdlib | Core concurrency model | Native to Python, provides the foundation for the low-latency pipeline. |
| **HTTP Client** | httpx | 0.27.x | Asynchronous API calls | High-performance `asyncio` client required for non-blocking API requests. |
| **Prompt Mgmt** | langchain-core | 0.2.x | Prompt templating | Provides robust `PromptTemplate` capabilities as required by the PRD. |
| **Configuration** | PyYAML | 6.x | Parsing `config.yaml` | Standard, reliable library for working with the chosen YAML config format. |
| **Voice Input** | SpeechRecognition| 3.10.x | Capturing microphone audio | A flexible library that can interface with multiple speech-to-text engines. |
| **LLM Service** | OpenAI | GPT-4 / GPT-5 | Core AI response generation| The chosen provider for high-quality, state-of-the-art language models. |
| **TTS Service** | ElevenLabs | v1 API | Text-to-Speech generation | The chosen provider for low-latency, high-quality voice synthesis. |
| **Testing** | pytest | 8.x | Unit & Integration testing | The de-facto standard for testing in Python; powerful and extensible. |

## 4\. Data Models

These are critical in-memory data structures that define the application's state.

  * **Application State:** Holds the real-time state of the performance (`current_status`, `current_phase`, `performance_plan`).
  * **Configuration:** Represents the settings loaded from `config.yaml` (API keys, model names, timings, etc.).
  * **Conversational Memory Turn:** Structures the conversation history for context (`user_input`, `ai_response`).
  * **Audio Queue Item:** Represents an audio file waiting to be played (`file_path`, `priority`).

## 5\. Components

  * **Initialization (`initialization.py`):** Loads and provides application settings from `config.yaml`.
  * **Input Handler:** Detects and interprets physical inputs, translating them into abstract application events.
  * **State Machine:** Manages the application's operational state (`IDLE`, `LISTENING`, etc.) and performance phase.
  * **Voice to Text (`voice_to_text.py`):** Captures and transcribes audio from the microphone.
  * **Decoder (`decoder_hadley.py`, etc.):** Translates the magician's coded phrases into secret data for the LLM.
  * **Conversational Memory (`conversational_memory.py`):** Stores and retrieves the conversation history.
  * **GPT Service (`gpt.py`):** Interfaces with the OpenAI LLM, formats prompts, and makes API calls.
  * **Text to Speech (`text_to_speech.py`):** Converts text into an audio file via the ElevenLabs API.
  * **Audio Queue (`audio_queue.py`):** Manages a priority queue of audio files for playback.
  * **Main Application (`main.py`):** The central orchestrator containing the `asyncio` event loop that connects all components.

## 6\. External APIs

  * **OpenAI API:** Used for LLM response generation (`POST /v1/chat/completions`) and potentially Speech-to-Text (`POST /v1/audio/transcriptions`).
  * **ElevenLabs API:** Used for low-latency TTS synthesis (`POST /v1/text-to-speech/{voice_id}/stream`).

## 7\. Core Workflows

This document includes sequence diagrams for the three primary workflows:

  * **"Sustained Input":** The main performance loop (Record -\> Decode -\> Respond).
  * **"Compound Input":** The direct-to-AI loop (Record -\> Respond, No Decode).
  * **"Brief Input":** The simple loop for playing the next item from the audio queue.

## 8\. REST API Spec

Not applicable. This application consumes external APIs but does not host its own.

## 9\. Database Schema

Not applicable. All application state is managed in-memory to ensure the lowest possible latency.

## 10\. Source Tree

```plaintext
corindagpt/
├── config/
│   ├── config.yaml
│   └── hadley_codes.json
├── docs/
│   ├── prd.md
│   └── architecture.md
├── prompts/
│   ├── phase_1_prompt.txt
│   └── ...
├── src/
│   ├── __main__.py
│   ├── audio_queue.py
│   ├── components/
│   │   ├── input_handler.py
│   │   └── state_machine.py
│   ├── conversational_memory.py
│   ├── decoders/
│   │   └── decode_hadley.py
│   ├── services/
│   │   ├── gpt.py
│   │   ├── tts.py
│   │   └── voice_to_text.py
│   └── utils/
│       └── initialization.py
├── tests/
│   ├── test_state_machine.py
│   └── ...
├── .gitignore
├── README.md
└── requirements.txt
```

## 11\. Infrastructure and Deployment

The application is deployed manually to a dedicated mini PC. The process involves cloning the Git repository, installing dependencies, configuring API keys, and setting up the main script to run on startup. Rollbacks are handled by checking out a previous, known-good Git tag.

## 12\. Error Handling Strategy

The strategy prioritizes maintaining the performance illusion. Critical errors, especially network/API failures, are caught by a global handler. Instead of a technical error message, the application will play a pre-recorded, in-character audio file stored locally (e.g., "I'm tired... I must leave you now.") and return to an idle state.

## 13\. Coding Standards

A minimal set of mandatory rules will be enforced, including: all I/O must be asynchronous; use a central configuration object; never hardcode secrets; use specific exception handling; and use the `logging` module instead of `print()`. Code will be formatted with **Black** and linted with **Flake8**.

## 14\. Test Strategy and Standards

A "Test-After Development" approach will be used with `pytest`, aiming for 85% line coverage on core logic. The strategy includes a large base of unit tests (mocking external APIs with `pytest-httpx`), a smaller set of integration tests, and mandatory manual performance validation on the target hardware.

## 15\. Security

The primary security focus is on protecting API keys in the `config.yaml` file on the mini PC through strict file system permissions. All external API calls must use HTTPS. The `pip-audit` tool will be used to scan for vulnerable dependencies.

