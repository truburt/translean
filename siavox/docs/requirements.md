# Requirements Document: Siavox

## 1. Document Overview

This document defines the high-level functional and non-functional requirements for **Siavox**, a mobile-optimized web application designed for speech transcription, automatic translation, and text transformation into structured notes or ready-to-send messages.

## 2. Product Overview

Siavox allows users to capture spoken language, automatically detect the source language, translate it if requested, and refine the raw transcript into a polished, concise note or message format. The system relies on a local infrastructure setup for data processing and AI inference.

## 3. Functional Requirements

### 3.1. User Authentication

* **Sign-in / Access Control:** The system must restrict access to authorized users only. Google Auth is the default.
* **Session Management:** The client interface must maintain a secure session state to prevent unauthorized access to the user's history.

### 3.2. Speech Processing Pipeline

* **Audio Capture:** The client must capture user speech or accept audio input directly through the interface.
* **Automated Language Detection:** The system must automatically detect the source language of the spoken input.
* **Transcription:** The system must convert the captured speech into accurate text.
* **Target Language Identification:** The system must automatically determine the target language for translation based on user instructions or explicit request. If nothing provided, target = source.
* **Text Transformation:** The processing pipeline must transform the raw transcript into one of two distinct formats based on user selection:
* **Note:** A clear, structured, and concise summary of the spoken thoughts.
* **Message:** A polished text ready for immediate dispatch/communication.

### 3.3. Client Interface (Mobile-First Web UI)

* **Responsive Design:** The web application UI must be fully optimized for mobile devices (smartphones and tablets).
* **Core Interaction Flows:**
* Triggering the transcription and transformation pipeline.
* Selecting the desired output format (**Note** vs. **Message**).
* Explicitly requesting translation when required.

* **One-Click Copy:** The UI must provide a dedicated button to copy the final transformed text to the device clipboard with a single tap.
* **History Viewer:** Users must be able to view a chronological log of their previously generated notes and messages.

### 3.4. History & Storage

* **Data Persistence:** The backend must securely store the history of processed interactions for each authorized user.
* **Retrieval:** Saved notes, messages, and transcripts must be fetchable by the client interface for historical review.

## 4. Behavioral & Processing Logic

1. The user authenticates and accesses the main interface.
2. The user initiates audio capture and speaks, optionally indicating if translation is required.
3. The backend receives the audio, detects the source language, and transcribes it.
4. The backend evaluates whether translation is necessary (detecting the target language automatically).
5. The backend routes the processed text through the local AI model to transform it into the requested format (Note/Message).
6. The final output is returned to the client, saved to history, and made available for instant copying.

## 5. Non-Functional & Infrastructure Requirements

### 5.1. Integration & AI Inference

* **Ollama Compatibility:** The backend processing cycle must interface with an Ollama instance already deployed within the local area network (LAN).
* **Model Constraints:** The system must utilize the `Gemma4 E4B it` model hosted on the local Ollama instance for all text transformation, language detection, and processing tasks.

### 5.2. Deployment

* **Containerization:** The entire application infrastructure (backend services, database/storage, and frontend distribution) must be configured to build and run using **Docker Compose**.
* **Network Isolation:** The setup must support local network operations, specifically targeting communication with the pre-existing local Ollama network endpoint.