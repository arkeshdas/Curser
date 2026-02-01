# Curser
**Arkesh Das**

## Motivation

One of my favorite hobbies is annoying my little sister. The issue is, now that I'm away from home, I've had to get more creative about how I go about this. My latest bit has been making increasingly more ridiculous nicknames for her and using only them to refer to her whenever I call her on the phone. This method is especially effective when the nickname is based on a nickname she already has, like the one my mom has for her.


And so the other day, that's I did. I started calling her a nickname over the phone. However, my sister was on speaker and within ear-shot of my mom. Normally, my mom either doesn't really care or she joins in so we can collectively annoy my sister, but this time she explicitly (and sternly) told me NOT call my sister that.


That's because, the nickname that I innocently gave her sounded *very close* to a slang word for a certain reproductive organ in Bengali.


I learned Bengali when I was a kid speaking with my parents and relatives, so unfortunately, the slang aspect of my lexicon is not up to par with the rest of my Bengali knowledge. On one hand, this was a great learning experience that allowed me to add a great linguistic tool to my sparse tool belt, but I also didn't want to run into this issue again where I spend all this brain power brainstorming a nickname, only for it get shot down.

## Overview

In its current state, this project is a (locally-run) interactive webpage/application that takes in spoken or typed imputs and finds the closest curse word in multiple languages to the input.


At a high level: the app records or accepts audio, transcribes speech locally using Whisper, converts candidate text spans into IPA phonemes with eSpeak, and compares them against the database of words and asks PanPhon to calculate distances from the input word/phrase to the db entries. Results then are ranked by phonetic similarity and presented with explanations, sortable tables, and optional text-to-speech playback on the web page.


Like I said before, its entirely local and supports live microphone input, one-shot recording, and audio uploads, with real-time level metering and session history tracking. The interface is built off of pure Streamlit for the interface, Whisper from OpenAI for speech recognition.

## Set-up

This project supports the `conda` environment setup. The `environment.yml` file contains all the project dependencies (as far as I know), so to be able to run this project, simply activate the conda environment.

To set up the `conda` environment, first make sure that you have a distribution of conda installed on your machine, such as Anaconda, Mini-Conda or Miniforge.

Then, pull this repository locally. 

Once you've done that, navigate to your remote clone of the repository and install the required dependencies using the command:

```
conda env create -f environment.yml
```

*Note: this may take a while, as there are a lot of packages that are not available in conda, so they are being pip-installed instead.*

Once the environment has been created, run this command to activate the environment:

```
conda activate curserio
```

Now, you should have all the packages to run the Curser application. Navigate to the **Quick Start** section for instructions on how to do that. 

# Structure

```
Curser/
│
├── app/
│   └── app.py                 # Main Streamlit application 
│
├── audios/
│   ├── coolio.m4a             # Sample audio input
│   ├── fudge(unclear).m4a     # Test audio with unclear pronunciation
│   └── Fudgeoff.m4a           # Additional audio sample
│
├── db/
│   ├── build_db.py            # Script to construct the phonetic database
│   ├── db_seed.json           # Seed data for initializing database
│   └── db.json                # Generated phonetic database

│
├── old_tests/
│   ├── db.py                  # Database access and helper functions
│   ├── demo_text.py           # Early prototype for text-based testing
│   └── test_soundalike.py     # Testing sound-alike word matching
│
├── scr/
│   ├── __pycache__/           # Cached Python bytecode
│   ├── __init__.py            # Package initializer
│   ├── asr.py                 # Automatic speech recognition logic
│   ├── core.py                # Core application logic
│   └── g2p.py                 # Grapheme-to-phoneme conversion module
│
├── .gitignore                 # Git ignore rules
├── environment.yml            # Conda environment specification
└── README.md                  # Project documentation
```

Currently, there are 5 main sub-directories.

The `app` folder contains the main `app.py` file, which contains the main app launcher and website structure. This is the piece that needs to be run to launch the app. See the **Quick Start** section below for more info.

The `audios` folder contains a few testing audios, so that if you chose to test the app by uploading a pre-recorded audio to the app. One is intentionally unclear, while the other two are meant to be slighly clearer test cases.

The `db` folder contains both the current database of words and a script that builds a data base given a `seed` json file. See the **Customizing the Word Database** for information on how to create your own database of words (in-case you want to use this project to do something useful). 

The `old_tets` folder contains old python code that I used to make sure the core parts of the app work. The app no longer needs these files to function.

The `scr` folder contains the actual meat of the app. In order for the app to recognize these python files and import them, I had to create an init file, but its empty. 

The `asr.py` file is used to determine the language if the user uses the auto language detect function. 

The `core.py` contains the core logic for the app. It's basically the scoring and phonetic matching engine of the project. It contains utility functions for tokenizing text, normalizing IPA, computing phonetic distances, ranking database candidates, and choosing the best word span from user input. Everything is organized into functions so they can be easily used seperately and debugged/modified if needed. 

The `g2p.py` file is what turns the text imput into a cleaned IPA (International Phonetic Alphabet) string using espeak, with special handling for multiple languages and Korean fallback logic.


## Quick Start

If you want to test out the application, run the `app.py` python file in a terminal window using the command:

```
streamlit run app/app.py
```

to launch the website. Make sure that you are running this command from the project root, NOT inside the app sub-folder, or any other sub folder, as that will cause Streamlit's pathfinding to fail. Example audio files can be found in the audios directory, or you can record your own audio from the app itself.

## Curstomizing the Word Database

So, if you want to use this app to do something actually useful (like do some basic speech recognition), you can modify the database of words that this app pulls from.

The `seed` file is much simpler than the main database, so it is much easier to add words to. It contains the categories for the word, the language, the meaning, the severity, and the category. This structure is general enough so that you could theoretically use it to recognize words for any purpose and rank them based off any quality just by changing what the severity field refers to.

To set up a new database, paste/create a file in the same simplified format as `db_seed.json` and run the `build_db.py` file to build the JSON database that the app will use. In order to actually use your newly generated database, rename the file to `db.json`. 

The output database adds 2 new categories and reformats the data so that the app scripts can read it. The two new categories are `ipa` and `ipa_norm`. These two categories are what are used to actually determine the closest word(s) to the input.

The `ipa` category is the `ipa` pronunciation of the word. This contains stress marks, tone indicators, and other subtle markers (I'm not a linguistics expert) that reflect closely how the word is pronounced in the standard pronunciation of whatever language the word comes from.

Unfortunately, this alone does not guarantee accurate analysis, since there are various ways that words can be pronounced. This problem is usually addressed in larger (and more complicated) language detection models by encoding multiple `ipa` pronunciations, however, for the purposes of this app, the best compromise that I found was just using a category called `ipa_norm`, which is essentially the IPA pronunciation without the additional pronunciation markers. 

## Challenges I ran into


## What I learned


## What's next for Curser