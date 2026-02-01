# Curser
**Arkesh Das**

## Motivation

One of my favorite hobbies is annoying my little sister. The issue is, now that I'm away from home, I've had to get more creative about how I go about this. My latest bit has been making increasingly more ridiculous nicknames for her and using only them to refer to her whenever I call her on the phone. This method is especially effective when the nickname is based on a nickname she already has, like the one my mom has for her.


And so the other day, that's I did. I started calling her a nickname over the phone. However, my sister was on speaker and within ear-shot of my mom. Normally, my mom either doesn't really care or she joins in so we can collectively annoy my sister, but this time she explicitly (and sternly) told me NOT call my sister that.


That's because, the nickname that I innocently gave her sounded *very close* to a slang word for a certain reproductive organ in Bengali.

I learned Bengali when I was a kid speaking with my parents and relatives, so unfortunately, the slang aspect of my lexicon is not up to par with the rest of my Bengali knowledge. On one hand, this was a great learning experience that allowed me to add a great linguistic tool to my sparse tool belt, but I also didn't want to run into this issue again where I spend all this brain power brainstorming a nickname, only for it get shot down.

## Why This App Exists: Because Bad Names Are Forever

It turns out that coming up with names is hard, especially when you want them to be unique, abstract, and not accidentally offensive in some other language. A bad name doesn't just tell potential customers and investors that you don't care, it quietly puts you at a disadvantage before anyone even looks at what you built.

Pharmaceutical companies deal with this exact problem constantly. When creating brand names for drugs, they are required to use names that are clearly distinct from existing medications and are not allowed to imply efficacy, mechanism, or outcomes. You cannot call a drug something that sounds like it cures cancer, even if you really wish it did. This pushes companies toward strange, abstract, made up names that are legally safe, but often linguistically fragile once the drug is sold overseas. A name that passed every legal review can still fall apart the moment it hits another language or culture.

The same problem shows up in tech, just with fewer lawyers involved and more optimism. If you are at a hackathon, YC, or convincing yourself that your next LLM wrapper is definitely the next big thing, you probably want a name that sounds catchy and distinctive. Names like Hulu, Zillow, Spotify, Stripe, or Airbnb work because they are abstract enough to stand out. That abstraction is also a risk, because you are always one unlucky phonetic collision away from naming your product something unintentionally cursed somewhere else. At that point, it does not matter how good the idea is, the damage is already done.

Curser sits right in that uncomfortable gap. It is a tool for answering the question, “Does this innocent sounding word get me in trouble somewhere else?”, before you commit to it as a nickname, product name, brand, or inside joke.

In other words, this is a nickname QA tool that doubles as a way to avoid very preventable naming regret.


## Overview

In its current state, this project is an interactive web application that takes spoken or typed input and identifies the closest potentially offensive or curse-like words across multiple languages based on phonetic similarity.

At a high level, the app records or accepts audio, transcribes speech locally using Whisper, converts candidate text spans into IPA phonemes, and compares them against a curated multilingual database of words. PanPhon is used to compute phonetic distances between the input and database entries, allowing results to be ranked by similarity and surfaced through an interactive interface with explanations, sortable tables, and session-level history tracking.

In addition to analysis, Curser supports optional text-to-speech playback of matched words. By default, pronunciation is handled locally using eSpeak, but the app can also integrate ElevenLabs for higher-quality, neural voice synthesis when enabled. This makes it possible to hear how flagged words actually sound across different languages and voices, which is often critical when evaluating phonetic ambiguity in spoken contexts.

The entire pipeline runs locally by default and supports live microphone input, one-shot recording, and audio uploads, complete with real-time level metering. The interface is built entirely with Streamlit, while speech recognition is powered by Whisper. Together, these components make Curser a practical tool for detecting accidental phonetic collisions across languages, especially when evaluating names, brands, or terminology that must avoid unintended or offensive interpretations.

## Set-up

This project uses a standard Python virtual environment with pip-based dependency management.

To get started, first make sure you have a recent version of Python installed (Python 3.10 or newer is recommended). Then, clone this repository locally.

From the root of the repository, create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate   (for macOS/Linux)
```
Next, install the required dependencies:
```
pip install -r requirements.txt
```

Now, you should have all the packages to run the Curser application locally . Navigate to the **Quick Start** section for instructions on how to do that. 

### Optional: ElevenLabs text-to-speech integration

Curser supports optional neural text-to-speech playback using ElevenLabs. This is not required for the core phonetic analysis pipeline. If no ElevenLabs API key is provided, the app will fall back to local eSpeak-based pronunciation.

To enable ElevenLabs voice output:
1.	Create an ElevenLabs account and generate an API key.
2.	Add the key to Streamlit secrets.

For local development, create a file at:
```
.streamlit/secrets.toml
```
and add:
```
ELEVENLABS_API_KEY = "your_api_key_here"
```
When an API key is present, an additional toggle will appear in the interface allowing you to switch between local eSpeak voices and ElevenLabs neural voices for pronunciation playback.

The ElevenLabs integration is intended for demonstration and qualitative evaluation of phonetic ambiguity, especially in spoken contexts. The core analysis, transcription, and scoring logic does not depend on ElevenLabs and runs fully locally by default.


## Structure

```
Curser/
│
├── app/
│   ├── app.py                 # Main Streamlit application
│   └── static/
│       └── curser-logo.png    # Application logo
│
├── audios/
│   ├── coolio.m4a             # Sample audio input
│   ├── fudge(unclear).m4a     # Test audio with unclear pronunciation
│   └── Fudgeoff.m4a           # Additional audio sample
│
├── db/
│   ├── build_db.py            # Script to construct the phonetic database
│   ├── db_seed.json           # Seed data for initializing the database
│   └── db.json                # Generated phonetic database
│
├── old_tests/
│   ├── db.py                  # Early database helpers
│   ├── demo_text.py           # Early prototype for text-based testing
│   └── test_soundalike.py     # Initial sound-alike matching experiments
│
├── src/
│   ├── __init__.py            # Package initializer
│   ├── asr.py                 # Speech recognition and language handling
│   ├── core.py                # Core phonetic matching and scoring logic
│   └── g2p.py                 # Grapheme-to-phoneme conversion utilities
│
├── requirements.txt           # pip-based Python dependencies
├── packages.txt               # System-level dependencies for deployment
├── runtime.txt                # Python runtime specification
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

### Directory Overview

Currently, there are five main sub-directories.

The `app` directory contains the Streamlit application entry point (`app.py`) along with static assets such as the application logo. Running this file launches the Curser web interface For more information on how to run the app, see the **Quick Start** section below.

The `audios` directory includes sample audio files used to test transcription and phonetic matching. These range from clearly spoken inputs to intentionally ambiguous pronunciations for stress-testing the pipeline.

The `db` directory contains the phonetic database and supporting scripts. `db_seed.json` defines the raw word entries, while `build_db.py` converts them into IPA representations and produces the final searchable database (`db.json`). See the **Customizing the Word Database** for information on how to create your own database of words.

The `old_tests` directory contains legacy scripts and early prototypes that were used to validate core ideas during development. These files are no longer required for the application to function.

The `src` directory holds the core logic of the project.
- `asr.py` (Automatic Speech Recognition) handles speech transcription and language inference.
- `core.py` implements the phonetic matching engine, including tokenization, IPA normalization, distance computation, and result ranking.
- `g2p.py` (Grapheme-to-Phoneme) converts text into cleaned IPA using eSpeak, with multilingual support and fallback handling.

The remaining files (`requirements.txt`, `packages.txt`, and `runtime.txt`) define the Python and system dependencies needed for local execution and deployment. See the **Set-up** section below for directions on how to install the dependencies needed for the project to run.
 

## Quick Start

If you want to test out the application, run the `app.py` Python file from a terminal using:

```
streamlit run app/app.py
```

This launches the Streamlit web interface. Make sure you run this command from the project root, not from inside the `app/` directory or any other subfolder, otherwise Streamlit’s path resolution will fail.

Example audio files are provided in the audios/ directory, but you can also record live audio or upload your own recordings directly through the app interface.


## Customizing the Word Database

While the default database is focused on identifying curse words across languages, the underlying system is intentionally more general. You can adapt it for lightweight language detection, phonetic screening, name vetting, brand safety checks, or any task where “what this sounds like” matters more than how it is spelled.

### Seed database structure

The database is defined in two stages. The first is a simplified seed file, such as `db_seed.json`. This file is intentionally human-readable and easy to edit. Each entry includes fields like:
- `word`: the canonical form used for phonetic matching
- `display`: the version shown in the UI (optional but recommended)
- `lang`: language code
- `meaning`: human-readable explanation
- `severity`: a numeric score you define
- `category`: any grouping label you want

The key distinction is between word and display:
- `word` is the form that gets converted to IPA and used internally for phonetic comparison
- `display` is what the user sees in the UI and result tables

This allows you to store words in a normalized or romanized form for consistent phonetic processing, while still displaying the correct native script, accents, or diacritics to users. For Latin-based languages, display can simply be the correctly accented spelling. For non-Latin scripts, this separation becomes especially important, since grapheme-to-phoneme behavior can vary by language, input encoding, and voice configuration, and phonetic consistency matters more than orthographic fidelity for sound-alike matching.

The `severity` field is intentionally abstract. In the default setup it represents offensiveness, but you can reinterpret it freely, for example as risk level, confidence score, priority, or any other ranking dimension relevant to your use case.

### Building the phonetic database

Once you have a seed file in the same format as `db_seed.json`, run:
```
python db/build_db.py
```
This script generates the full database that the app actually consumes. The output file includes two additional fields for each entry:
- `ipa`: the raw IPA transcription generated by eSpeak
- `ipa_norm`: a normalized version of the IPA string

The `ipa` field preserves stress marks, length markers, and other phonetic annotations that reflect a more precise pronunciation. However, exact IPA matches are often too strict for real-world audio, accents, and imperfect pronunciation.

To make matching more robust, the app relies primarily on `ipa_norm`, which strips out many of these markers and produces a more stable phonetic representation. This normalized form enables practical, cross-language sound-alike matching without requiring large neural models or multiple pronunciation variants per word.

After generation, rename the output file to `db.json` (if it is not already), and the app will automatically load it on the next run.

## Challenges I ran into

This project was way more friction-heavy than I initially expected, mostly because a lot of the difficulty lived in executing my vision rather than the core idea. 

Environment management was a constant source of problems.I initially accidentally installed all of the packages to my local conda installation, so I had to detangle everything from there. Then, once I got everything isolated into a conda environment, I realized that the streamlit application refused to cooperate with it, so I had to remake the entire thing using pip. System-level installs caused version conflicts that were hard to diagnose, especially when some packages were installed via distutils and could not be cleanly uninstalled. I ended up having to recreate environments multiple times just to get back to a known working state.

Audio and language tooling introduced another layer of complexity. Speech-to-text, G2P, and phonetic libraries all have slightly different assumptions about language codes, encodings, and input formats. Small mismatches, like language auto-detection behaving differently than expected or IPA output formats not lining up across tools, caused silent failures or confusing outputs that took time to trace back to the source.

Debugging was also harder than usual because many failures were not hard crashes. Instead, things would run but produce empty outputs, incorrect matches, or misleading results. That made it difficult to tell whether the issue was with my logic, the library behavior, or the input data itself. A lot of progress came from adding explicit checks, printing intermediate outputs, and simplifying the pipeline to isolate where things were breaking.

Finally, scope control was a challenge. It was tempting to keep adding features, like live mic input or better ranking heuristics, before the core pipeline was fully stable. In hindsight, locking down a minimal, reproducible version earlier would have saved time and reduced mental overhead while debugging.

## What I learned

This project was a much bigger learning experience than I initially expected. I learned a lot about language and phonetics. I have been interested in linguistics since taking an anthropology class in high school, but I had no formal training going into this. Working with IPA, grapheme-to-phoneme systems, and cross-language comparisons forced me to think more precisely about how spoken language is represented and where different tools succeed or fail.

This was also my first time deploying a web application, and my first project where the explicit goal was to make the code safe and reproducible so that other people could run it locally (learned about virtual environments 2 weeks ago in one of my data science classes). Managing virtual environments, dependencies, and setup instructions showed me how easily projects can break without careful environment control, and why reproducibility is a core part of real software development.

I also learned a lot about working with third-party APIs through integrating ElevenLabs for text-to-speech. This was my first time using an API that required authentication via a private token, and it forced me to think more carefully about security and configuration. I learned how to manage API keys without hard-coding them into the repository, including using Streamlit’s secrets.toml system and environment-specific configuration. More generally, working with the ElevenLabs API helped me understand how external services expose functionality, how client libraries can return streamed or chunked data instead of simple values, and how small interface assumptions can break an application if they are not handled carefully.

I also learned how to independently manage and iterate on a large project. Working alone, I had to define a minimal viable product, get a full pipeline working, then incrementally add new technologies and features while breaking and fixing things along the way. This process taught me how to scope work realistically, debug systematically, and prioritize stability and clarity over adding features too quickly.


## What's next for Curser

- **Ship a proper hosted version.** Right now it works great locally and hosted on streamlit's cloud servers, but a deployed version with a real backend would make it easier for other people to try. A more dynamic frontend (React) could also make the live mic experience smoother and less hacky than Streamlit reruns.

- **Make scoring customizable instead of “one size fits all.”** I want to expose the knobs behind the matching: weighting vowels vs consonants, rewarding same syllable counts, penalizing stress mismatches, preferring longer spans, or applying language-specific penalties. This would let users tune Curser for different naming scenarios (drug names vs startups vs product features).

- **Explain results better**. The distance score is useful, but I want to show more interpretability: which phoneme substitutions mattered most, where the best window alignment was found, and why one candidate outranked another.

- **Expand and maintain the database.** Long-term, I’d like a repeatable pipeline for adding new languages and maybe regional dialects from sources cleanly.

- **Automate name checks from real workflows**. A simple “paste a list of candidate names” mode, plus a CSV export, would make it feel more like an actual tool. A stretch goal is integrating lightweight web scraping, like pulling candidate names from a page or a doc, but only if it supports a real use case and doesn’t turn into a brittle gimmick.


## Data Sources and Credits

The profanity database used in this project is compiled from multiple sources. One major source of seed words comes from the following repository:

https://github.com/4troDev/profanity.csv/tree/main

I did not use this dataset as is.

To make it usable for phonetic matching, I pulled the raw word lists and removed compound or multi word phrases. This was necessary because PanPhon operates at the phoneme level and compound phrases tended to produce misleading distance calculations that dominated the rankings for the wrong reasons.

After filtering, the remaining words were scraped and converted into simplified seed files. These seed files were then processed through eSpeak to generate IPA representations, normalized IPA variants, and finally compiled into the phonetic database used by the app.