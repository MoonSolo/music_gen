# music_gen
AI generated music on demand. Running locally with GPU/CPU.
=======
\# MusicGen Pipeline — README



This tool lets you automatically generate hours of AI music, stitch it together, and combine it with a looping video to create YouTube-ready videos. No prior technical knowledge is required — just follow this guide from top to bottom.



---



\## What you'll need



\- A Windows 10 or 11 computer

\- An NVIDIA GPU (recommended: RTX 3060 or better)

\- At least 16 GB of RAM

\- At least 20 GB of free disk space

\- An internet connection for the initial setup



---



\## Part 1 — First-time setup (do this once)



\### Step 1 — Install WSL2 (Ubuntu on Windows)



WSL2 lets you run Linux commands directly inside Windows. This is required.



1\. Open \*\*PowerShell as Administrator\*\*

&nbsp;  - Press the Windows key, type `PowerShell`

&nbsp;  - Right-click it and choose \*\*Run as administrator\*\*



2\. Run this command:

&nbsp;  ```

&nbsp;  wsl --install

&nbsp;  ```



3\. When it finishes, \*\*restart your computer\*\*.



4\. After restart, a black Ubuntu window will open automatically. It will ask you to create a username and password. Choose anything you like — \*\*remember these\*\*, you'll need them later.



5\. Verify it worked by running:

&nbsp;  ```

&nbsp;  wsl --list --verbose

&nbsp;  ```

&nbsp;  You should see \*\*Ubuntu\*\* listed with \*\*Version 2\*\*.



---



\### Step 2 — Install Visual Studio Code (recommended)



VS Code makes it much easier to edit files and run commands.



1\. Download from: https://code.visualstudio.com/download

2\. Install it normally

3\. Open VS Code, go to Extensions (`Ctrl+Shift+X`), search for \*\*WSL\*\* and install it (by Microsoft)

4\. Also install the \*\*Python\*\* extension (by Microsoft)



---



\### Step 3 — Run the setup script



1\. Open your Ubuntu terminal

&nbsp;  - Press the Windows key, type `Ubuntu`, and open it



2\. Download the setup script:

&nbsp;  ```bash

&nbsp;  curl -o ~/setup.sh https://raw.githubusercontent.com/your-repo/setup.sh

&nbsp;  ```

&nbsp;  > \*\*Note:\*\* If you downloaded `setup.sh` manually, copy it into your Ubuntu home folder instead.

&nbsp;  > From Windows, your Ubuntu home folder is accessible at:

&nbsp;  > `\\\\wsl$\\Ubuntu\\home\\YOUR\_USERNAME\\`



3\. Make it executable and run it:

&nbsp;  ```bash

&nbsp;  chmod +x ~/setup.sh

&nbsp;  ./setup.sh

&nbsp;  ```



4\. The script will take \*\*10–20 minutes\*\* to complete (it downloads several GB of AI model dependencies). You will see progress printed for each step.



5\. At the end you should see something like:

&nbsp;  ```

&nbsp;  ✓ PyTorch   : 2.5.1+cu121

&nbsp;  ✓ GPU       : True

&nbsp;  ✓ Device    : NVIDIA GeForce RTX 3060

&nbsp;  ✓ AudioCraft: OK

&nbsp;  ✓ Everything is installed and ready.

&nbsp;  ```



If you see any red error messages, please refer to the \*\*Troubleshooting\*\* section at the bottom of this file.



---



\### Step 4 — Open the project in VS Code



1\. In your Ubuntu terminal:

&nbsp;  ```bash

&nbsp;  cd ~/musicgen\_pipeline

&nbsp;  code .

&nbsp;  ```



2\. VS Code will open connected to your Ubuntu environment. You'll see \*\*WSL: Ubuntu\*\* in the bottom-left corner.



3\. Open the integrated terminal inside VS Code with `` Ctrl+` `` and activate the environment:

&nbsp;  ```bash

&nbsp;  source venv/bin/activate

&nbsp;  ```

&nbsp;  You should see `(venv)` appear at the start of the line. \*\*Do this every time you open a new terminal.\*\*



---



\## Part 2 — Creating your prompt file



The prompt file tells the pipeline what music to generate. It lives at:

```

~/musicgen\_pipeline/prompts.txt

```



A sample file was created automatically by the setup script. Open it in VS Code to edit it.



\### Format



Each line follows this structure:

```

description of the music | duration in seconds | filename

```



\*\*Example:\*\*

```

lofi hip hop, relaxed, 85 BPM, vinyl texture, rain sounds | 30 | lofi\_001

dark ambient, cinematic, slow, deep bass, tension | 45 | ambient\_001

upbeat jazz, piano, trumpet, 120 BPM, energetic | 30 | jazz\_001

```



\### Rules

\- The three parts are separated by `|`

\- Duration must be between \*\*5 and 360 seconds\*\*

\- Filename must be unique (no duplicates in the same file)

\- Lines starting with `#` are comments and are ignored

\- Blank lines are ignored



\### Tips for good prompts

\- Be specific: `lofi hip hop, 85 BPM, rainy night, vinyl crackle` works better than just `lofi`

\- Add mood words: `melancholic`, `energetic`, `peaceful`, `tense`, `dreamy`

\- Add instrument names: `piano`, `guitar`, `saxophone`, `synth pads`, `drums`

\- Vary your prompts slightly between clips of the same genre to avoid repetition



\### Clip duration advice

\- \*\*30 seconds\*\* per clip is the sweet spot for quality

\- Clips are automatically grouped into tracks of 2–6 minutes each

\- Aim for enough total duration to reach 1–2 hours of audio (120–240 clips of 30s)



---



\## Part 3 — Running the pipeline



Make sure you are in the project folder with the venv activated:

```bash

cd ~/musicgen\_pipeline

source venv/bin/activate

```



\### Option A — Full pipeline (generate + stitch + assemble video)



```bash

python3 scripts/run.py --model large --prompts prompts.txt --video\_name my\_video --animation assets/loop.mp4

```



This will generate all clips, stitch them together, and combine with your animation into a final MP4.



\### Option B — Audio only (no animation yet)



```bash

python3 scripts/run.py --model large --prompts prompts.txt --video\_name my\_video

```



This generates and stitches the audio. You can add the video later (see Option D).



\### Option C — Resume an interrupted run



If the pipeline was stopped or crashed, just run the exact same command again. It will automatically skip clips that were already generated.



\### Option D — Assemble video after the fact



Once you have your animation ready:

```bash

python3 scripts/run.py --model large --prompts prompts.txt --video\_name my\_video --skip\_generate --skip\_stitch --animation assets/loop.mp4

```



---



\## Choosing the right model



| Model | Quality | Speed | GPU VRAM needed |

|-------|---------|-------|-----------------|

| `small` | Good | Fastest | 2 GB |

| `medium` | Better | Moderate | 6 GB |

| `large` | Best | Slowest | 12 GB |



\*\*RTX 3060 (12 GB):\*\* Use `--model large` for best results.



\*\*Older or weaker GPU (4 GB):\*\* Use `--model small`.



\*\*No GPU / want to use CPU:\*\*

```bash

python3 scripts/run.py --model medium --device cpu --prompts prompts.txt --video\_name my\_video

```

Note: CPU generation is significantly slower (20–40 minutes per clip).



---



\## Expected generation times



For a 1-hour video (120 clips × 30 seconds):



| Setup | Time |

|-------|------|

| RTX 3060, large model, GPU | ~2–3 hours |

| RTX 3060, medium model, GPU | ~1–1.5 hours |

| Any machine, small model, GPU | ~30–60 minutes |

| Any machine, medium model, CPU | ~40–70 hours |



Stitching and video assembly add only a few minutes on top.



---



\## Output folder structure



After running, your files will be organized like this:



```

musicgen\_pipeline/

&nbsp; generated/

&nbsp;   my\_video/

&nbsp;     my\_video\_part\_001/

&nbsp;       lofi\_001.wav

&nbsp;       lofi\_002.wav

&nbsp;       my\_video\_part\_001\_stitched.mp3   ← 2–6 min track

&nbsp;     my\_video\_part\_002/

&nbsp;       ...

&nbsp;       my\_video\_part\_002\_stitched.mp3

&nbsp;     my\_video\_full.mp3                  ← full audio (1h+)

&nbsp;     my\_video\_full.mp4                  ← final YouTube video

&nbsp;     progress.json                      ← resume tracker

```



---



\## Individual scripts reference



You can also run each script independently if needed.



\*\*generate.py\*\* — Generate clips only:

```bash

python3 scripts/generate.py --model large --prompts prompts.txt --video\_name my\_video

```



\*\*stitch.py\*\* — Stitch clips into tracks:

```bash

python3 scripts/stitch.py --video\_name my\_video --crossfade 3

```



\*\*assemble.py\*\* — Assemble final video:

```bash

python3 scripts/assemble.py --video\_name my\_video --animation assets/loop.mp4

```



---



\## Troubleshooting



\*\*`(venv)` is not showing in my terminal\*\*

You need to activate the virtual environment:

```bash

cd ~/musicgen\_pipeline

source venv/bin/activate

```



\*\*`nvidia-smi` shows no GPU inside Ubuntu\*\*

Make sure your NVIDIA drivers are up to date on the Windows side (not inside Ubuntu). Download from: https://www.nvidia.com/Download/index.aspx



\*\*`torch.cuda.is\_available()` returns False\*\*

Re-run the setup script. If it still fails, check that your NVIDIA drivers are installed on Windows and that `nvidia-smi` works in the Ubuntu terminal.



\*\*Generation was killed / ran out of memory\*\*

You are trying to run a model that is too large for your hardware. Switch to a smaller model (`--model small`) or use CPU mode (`--device cpu`).



\*\*A clip failed but the rest continued\*\*

This is normal — failed clips are logged and skipped. You can check which ones failed in the terminal output. Re-run the same command to retry them.



\*\*ffmpeg not found\*\*

Run this in your Ubuntu terminal:

```bash

sudo apt install ffmpeg

```



\*\*The script says `No valid entries found`\*\*

Check your `prompts.txt` file. Make sure each line has exactly three parts separated by `|`, and that duration is a number between 5 and 360.



---



\## Quick reference card



```

\# Activate environment (do this every session)

cd ~/musicgen\_pipeline \&\& source venv/bin/activate



\# Full pipeline

python3 scripts/run.py --model large --prompts prompts.txt --video\_name NAME --animation assets/loop.mp4



\# Audio only

python3 scripts/run.py --model large --prompts prompts.txt --video\_name NAME



\# Resume interrupted run

python3 scripts/run.py --model large --prompts prompts.txt --video\_name NAME



\# Force CPU

python3 scripts/run.py --model medium --device cpu --prompts prompts.txt --video\_name NAME

```

tags :

lofi hip hop, lofi beats, chill beats, study music, lofi radio,
tokyo lofi, lofi japan, rainy lofi, night lofi, dark lofi,
3am lofi, insomnia lofi, lofi study, focus music, deep focus,
lofi playlist, lofi mix, beats to study to, beats to relax to,
rooftop lofi, sunset lofi, golden hour lofi, lofi chill,
background music, instrumental hip hop, late night lofi,
aesthetic lofi, sad lofi, moody lofi, lofi vibes,
ambient lofi, cozy lofi, relaxing music, lofi 2025