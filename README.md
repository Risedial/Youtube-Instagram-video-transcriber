# Video Whisper Transcriber

**Professional Windows desktop application for transcribing YouTube and Instagram videos to text using OpenAI Whisper AI**

![Platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Convert YouTube videos and Instagram Reels to accurate text transcriptions with 100% local processing—no cloud services, no subscriptions, complete privacy.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Technical Information](#technical-information)
- [Privacy and Legal](#privacy-and-legal)
- [Credits](#credits)

---

## Features

### Core Functionality

- **YouTube Video Transcription** - Convert any YouTube video (including Shorts) to text
- **Instagram Reels Transcription** - Transcribe Instagram Reels and video posts
- **100% Local Processing** - All transcription happens on your computer for complete privacy
- **Multiple Whisper Model Sizes** - Choose from tiny, base, small, medium, or large models
- **GPU and CPU Support** - Automatic GPU acceleration with CUDA-compatible graphics cards, or CPU fallback
- **Real-Time Progress Tracking** - Monitor download, extraction, and transcription progress

### Output Format Options

- **Timestamped Transcriptions** - Include timestamp markers for each segment (MM:SS format)
- **Plain Text Transcriptions** - Generate clean text without timestamps
- **Format Selection** - Choose your preferred output format via radio buttons in the UI

### User Interface

- **URL Input** - Paste YouTube or Instagram video URLs with automatic platform detection
- **Output Directory Selection** - Choose where transcription files will be saved
- **Whisper Model Selector** - Dropdown menu to select from 5 model sizes with performance information
- **Format Selection Radio Buttons** - Choose timestamp format (with or without)
- **Clear URL Button** - Clear the URL field while preserving all other settings
- **Start/Stop Controls** - Begin transcription or cancel mid-process with safe cleanup
- **Real-Time Status Display** - View detailed status messages and processing stages

### Advanced Features

- **Proxy/VPN Integration** - Configure proxy settings with anti-detection features
- **Multiple Proxy Types** - Support for HTTP, HTTPS, SOCKS4, and SOCKS5 proxies
- **Kill Switch** - Automatically stop operations if proxy connection fails
- **Anti-Detection Settings** - Random delays, user agent rotation, and header randomization
- **Comprehensive Error Handling** - Graceful error recovery with detailed user feedback
- **Platform Detection** - Automatic detection of YouTube vs Instagram URLs

### File Management

- **Configurable Output Directory** - Select and persist your preferred save location
- **Smart File Naming** - Files named based on video title and format (e.g., `video_title_with_timestamps.txt`)
- **Automatic Conflict Resolution** - Prevents overwriting existing files by appending numbers
- **Temporary File Cleanup** - Automatic cleanup of downloaded media files after transcription

---

## Prerequisites

### System Requirements

- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **RAM**:
  - Minimum: 4GB
  - Recommended: 8GB or more (required for larger Whisper models)
- **Storage**: Approximately 5GB for application and Whisper models
- **GPU** (Optional): CUDA-compatible NVIDIA GPU for faster processing
- **Internet Connection**: Required for downloading videos and Whisper models

### Software Dependencies

- **Python**: Version 3.11 or 3.12
- **FFmpeg**: Required for audio extraction (bundled in distributed executable version)

---

## Installation

> **Choose Your Path**:
> - **End Users**: Use the pre-built executable (easiest)
> - **Experienced Developers**: Quick setup from source
> - **Complete Beginners**: Detailed step-by-step guide below

### For End Users (Distributed Executable) --- COMING SOON ---

1. Download the latest release from the [Releases page](https://github.com/RiseDial/youtube-whisper-transcriber)
2. Extract the ZIP file to your desired location
3. Double-click `YouTube Whisper Transcriber.exe` to launch
4. On first run, the application will download the selected Whisper model
5. Configure your output directory and preferences

### For Experienced Developers (Quick Setup)

```bash
# Clone and navigate
git clone https://github.com/RiseDial/youtube-whisper-transcriber.git
cd youtube-whisper-transcriber

# Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Install FFmpeg (add to PATH)
# Download from: https://ffmpeg.org/download.html

# Run
python src\main.py
# or
run_app.bat
```

**Installation Time**: 15-30 minutes (mostly downloading dependencies ~3GB)

---

### For Complete Beginners (Detailed Guide)

**⏱️ Total Time**: 20-40 minutes
**💾 Download Size**: ~3GB
**📦 Disk Space**: ~5GB total (including models)

Follow these steps carefully. Each step includes verification to ensure everything is working.

#### Step 1: Install Python 3.11 or 3.12

1. **Download Python**
   - Go to [python.org/downloads](https://www.python.org/downloads/)
   - Download Python 3.11 or 3.12 (Windows installer, 64-bit)

2. **Run the Installer**
   - **⚠️ CRITICAL**: Check the box **"Add Python to PATH"** at the bottom of the first screen
   - Click "Install Now"
   - Wait for installation to complete
   - Click "Close" when done

3. **Verify Installation**
   - Open Command Prompt (Press `Windows Key`, type `cmd`, press Enter)
   - Type: `python --version`
   - You should see: `Python 3.11.x` or `Python 3.12.x`
   - If you see an error, Python wasn't added to PATH correctly—reinstall with the checkbox checked

#### Step 2: Install FFmpeg

FFmpeg is required for extracting audio from videos.

1. **Download FFmpeg**
   - Go to: [https://github.com/BtbN/FFmpeg-Builds/releases](https://github.com/BtbN/FFmpeg-Builds/releases)
   - Download: `ffmpeg-master-latest-win64-gpl.zip` (scroll down to Assets)
   - Extract the ZIP file to `C:\ffmpeg` (create this folder if it doesn't exist)

2. **Add FFmpeg to System PATH**
   - Press `Windows Key` and type: `environment variables`
   - Click "Edit the system environment variables"
   - Click the "Environment Variables..." button (bottom right)
   - Under "System variables" (bottom section), find and select "Path"
   - Click "Edit..."
   - Click "New"
   - Type: `C:\ffmpeg\bin`
   - Click "OK" on all windows

3. **Verify FFmpeg Installation**
   - **Close and reopen** Command Prompt (important!)
   - Type: `ffmpeg -version`
   - You should see FFmpeg version information
   - If you see `'ffmpeg' is not recognized`, the PATH wasn't set correctly—repeat step 2

#### Step 3: Download the Repository

1. **Install Git** (if not already installed)
   - Download from: [git-scm.com/downloads](https://git-scm.com/downloads)
   - Run installer with default settings
   - Restart Command Prompt after installation

2. **Clone the Repository**
   - Open Command Prompt
   - Navigate to where you want the project (e.g., Documents):
     ```bash
     cd Documents
     ```
   - Clone the repository:
     ```bash
     git clone https://github.com/RiseDial/youtube-whisper-transcriber.git
     ```
   - Navigate into the folder:
     ```bash
     cd youtube-whisper-transcriber
     ```

#### Step 4: Create Virtual Environment

A virtual environment keeps project dependencies separate from your system Python.

1. **Create the Environment**
   ```bash
   python -m venv venv
   ```
   - This takes 30-60 seconds
   - You'll see a new `venv` folder appear

2. **Activate the Environment**
   ```bash
   venv\Scripts\activate
   ```
   - You should see `(venv)` appear at the start of your command prompt
   - This means the virtual environment is active

#### Step 5: Install Dependencies

This step downloads and installs all required libraries (~3GB, takes 15-30 minutes).

1. **Install All Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   - **Be patient**: This downloads PyTorch, Whisper, and many other libraries
   - **Do not close** Command Prompt while this runs
   - You'll see progress messages as packages install
   - If you see errors, see [Common Installation Issues](#common-installation-issues) below

2. **Wait for Completion**
   - When finished, you'll see your command prompt return
   - Should show: `Successfully installed [list of packages]`

#### Step 6: Run the Application

1. **Make Sure Virtual Environment is Active**
   - Your command prompt should show `(venv)` at the start
   - If not, run: `venv\Scripts\activate`

2. **Launch the Application**

   **Option A: Using the batch file (recommended)**
   ```bash
   run_app.bat
   ```
   - The batch file automatically activates the virtual environment

   **Option B: Direct Python command**
   ```bash
   python src\main.py
   ```

3. **First Run**
   - The application window should open
   - On first transcription, Whisper will download the selected model (39MB to 1.5GB)
   - This is a one-time download per model

#### Step 7: Verify Everything Works

1. **Test with a Short Video**
   - Find a short YouTube video (1-2 minutes)
   - Copy the URL
   - Paste it into the application
   - Select "tiny" model (fastest, smallest)
   - Click "Start Transcription"
   - Wait for completion

2. **Success!**
   - If transcription completes, everything is installed correctly
   - The output file will be in your selected output directory

---

### Installation Verification Checklist

After installation, verify each component:

- [ ] **Python installed**: `python --version` shows 3.11 or 3.12
- [ ] **FFmpeg installed**: `ffmpeg -version` shows version info
- [ ] **Virtual environment active**: Command prompt shows `(venv)`
- [ ] **Dependencies installed**: `pip list` shows torch, whisper, yt-dlp, etc.
- [ ] **Application launches**: `python src\main.py` opens the GUI window
- [ ] **Test transcription works**: Successfully transcribes a short test video

---

### Common Installation Issues

#### Issue: `'python' is not recognized as an internal or external command`

**Cause**: Python wasn't added to PATH during installation

**Solution**:
1. Uninstall Python (Windows Settings → Apps → Python)
2. Download Python installer again
3. Run installer and **CHECK** the "Add Python to PATH" box
4. Complete installation
5. Restart Command Prompt

---

#### Issue: `'ffmpeg' is not recognized as an internal or external command`

**Cause**: FFmpeg isn't in system PATH

**Solution**:
1. Verify FFmpeg is at `C:\ffmpeg\bin\ffmpeg.exe`
2. If not, extract ZIP to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to PATH:
   - Press Windows Key → type "environment"
   - Edit system environment variables
   - Environment Variables → System variables → Path → Edit
   - New → `C:\ffmpeg\bin` → OK
4. **Restart Command Prompt** (critical step!)
5. Test: `ffmpeg -version`

---

#### Issue: `pip install` fails with "Could not find a version that satisfies the requirement torch"

**Cause**: Network issues or pip cache problems

**Solutions**:
1. **Update pip first**:
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Try again**:
   ```bash
   pip install -r requirements.txt
   ```

3. **If still fails**, install PyTorch separately first:
   - Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
   - Select: Windows, Pip, Python, CPU (or CUDA if you have NVIDIA GPU)
   - Copy and run the command shown
   - Then run: `pip install -r requirements.txt`

---

#### Issue: Virtual environment won't activate (PowerShell users)

**Cause**: PowerShell execution policy restrictions

**Solution**:
1. Open PowerShell as Administrator
2. Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
3. Type `Y` to confirm
4. Close PowerShell
5. Try activating again: `venv\Scripts\activate`

**Alternative**: Use Command Prompt (cmd) instead of PowerShell

---

#### Issue: Installation takes too long or appears stuck

**Expected Behavior**:
- Installing dependencies typically takes 15-30 minutes
- PyTorch alone is ~800MB and takes time
- You should see progress messages

**What to Do**:
- Be patient—as long as you see occasional progress, it's working
- Do not close the window
- If truly stuck (no activity for 10+ minutes), press `Ctrl+C`, then try again

---

#### Issue: Out of disk space during installation

**Cause**: Insufficient storage for dependencies

**Solution**:
1. Free up at least 5GB of disk space
2. Delete the `venv` folder
3. Run installation again: `python -m venv venv` and continue from Step 4

---

#### Issue: Application won't start - "Import Error" or "No module named..."

**Cause**: Virtual environment not activated or dependencies not fully installed

**Solution**:
1. Ensure virtual environment is active: `venv\Scripts\activate`
2. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
3. Verify installation: `pip list` should show whisper, torch, yt-dlp, etc.

---

### Getting More Help

If you encounter issues not covered here:

1. **Check the error message carefully** - it often explains what's wrong
2. **Search the error on Google** - many installation issues have known solutions
3. **Open an issue on GitHub** - [Report your problem](https://github.com/RiseDial/youtube-whisper-transcriber/issues)
4. Include:
   - Your Python version: `python --version`
   - Your operating system
   - The full error message
   - What step you're on

---

## Usage

### Quick Start Guide

#### Step 1: Launch the Application

- **Executable version**: Double-click `YouTube Whisper Transcriber.exe`
- **Source version**: Run `run_app.bat` or `python src\main.py`

The main window opens with a clean, professional interface.

#### Step 2: Enter Video URL

1. Paste a YouTube or Instagram Reel URL into the **Video URL** field
2. The application automatically detects the platform (YouTube or Instagram)
3. Valid URLs show a green confirmation message

**Supported URLs:**
- YouTube: `https://www.youtube.com/watch?v=...` or `https://youtu.be/...`
- Instagram: `https://www.instagram.com/reel/...` or `https://www.instagram.com/p/...`

#### Step 3: Select Output Directory

1. Click **Browse** next to the Output Directory field
2. Choose where you want transcription files saved
3. This setting is saved and will persist across sessions

#### Step 4: Choose Whisper Model

Select a model from the dropdown based on your priorities:

| Model | Size | Speed | Accuracy | Memory | Best For |
|-------|------|-------|----------|--------|----------|
| **tiny** | 39MB | Fastest (~32x realtime) | Good | 1GB | Quick drafts, testing |
| **base** | 74MB | Fast (~16x realtime) | Better | 1GB | General use |
| **small** | 244MB | Medium (~6x realtime) | Good | 2GB | Balanced quality/speed |
| **medium** | 769MB | Slow (~2x realtime) | Very Good | 5GB | High accuracy needs |
| **large** | 1550MB | Slowest (~1x realtime) | Best | 10GB | Maximum accuracy |

**Note**: On first use of each model, it will be downloaded and cached (one-time process).

#### Step 5: Select Timestamp Format

Choose your preferred output format using the radio buttons:

- **Include Timestamps**: Generates transcription with timestamp markers
  - Example: `[00:15 - 00:20] Welcome to this tutorial`
- **Exclude Timestamps**: Generates plain text transcription without timestamps
  - Example: `Welcome to this tutorial`

Only one format can be selected at a time.

#### Step 6: Start Transcription

1. Click **Start Transcription**
2. Monitor the progress bar showing current stage:
   - Validation → Download → Transcription → Saving
3. View detailed status messages in the status display
4. Wait for completion (time varies by video length and model size)

**Keyboard Shortcut**: `Ctrl+Enter`

#### Step 7: Locate Output Files

When complete, transcription files are saved to your selected output directory:

- **With timestamps**: `[video_title]_with_timestamps.txt`
- **Without timestamps**: `[video_title]_without_timestamps.txt`

The application will prompt to open the output folder.

---

### Advanced Features

#### Proxy/VPN Settings

Access via **Settings** → **Proxy/VPN Settings...**

Configure proxy protection for:
- Bypassing geographic restrictions
- IP address protection
- Rate limit avoidance

**Available Settings:**

1. **Proxy Server Configuration**
   - Proxy type: HTTP, HTTPS, SOCKS4, or SOCKS5
   - Host and port configuration
   - Optional username/password authentication

2. **Anti-Detection Features**
   - Random delays between requests (configurable range in seconds)
   - User agent rotation
   - HTTP header randomization

3. **Kill Switch**
   - Automatically stops all operations if proxy connection fails
   - Protects against IP leakage
   - Enabled by default

4. **Connection Testing**
   - Test proxy connection before use
   - Verify configuration without starting transcription

#### UI Controls Reference

**Clear URL Button**
- Clears only the URL input field
- Preserves output directory setting
- Preserves Whisper model selection
- Preserves timestamp format choice
- Preserves all proxy settings

**Stop Button**
- Safely cancels current transcription
- Performs cleanup of temporary files
- Returns to ready state
- Can start new transcription immediately after stopping

**Format Radio Buttons**
- Mutually exclusive selection (only one active at a time)
- Selection is saved automatically
- Persists across application restarts

#### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Start transcription |
| `Escape` | Stop processing |
| `Ctrl+Delete` | Clear URL field |
| `F1` | Show help |
| `Tab` | Navigate between fields |
| `Alt+F4` | Close application |

---

## Output Files

### File Naming Conventions

The application generates files based on the video title and selected format:

**With Timestamps:**
```
video_title_with_timestamps.txt
```

**Without Timestamps:**
```
video_title_without_timestamps.txt
```

**File Conflict Handling:**
If a file already exists, the application automatically appends a number:
```
video_title_with_timestamps_1.txt
video_title_with_timestamps_2.txt
```

### File Structure

#### Timestamped Format Example

```
# Transcription Generated by YouTube/Instagram Video Transcriber
# Generated on: 2024-01-15 14:30:00
# Language: en
# Model: base
# Processing time: 45.2 seconds
# Audio duration: 180.5 seconds
# Word count: 1,234

--------------------------------------------------

# Full Transcription

Welcome to this tutorial on machine learning fundamentals...

--------------------------------------------------

# Transcript with Timestamps

[00:00 - 00:05] Welcome to this tutorial on machine learning fundamentals
[00:05 - 00:12] In this video, we will discuss the key concepts you need to know
[00:12 - 00:18] Let's start with supervised learning algorithms
```

#### Plain Text Format Example

```
# Transcription Generated by YouTube/Instagram Video Transcriber
# Generated on: 2024-01-15 14:30:00
# Language: en
# Model: base
# Word count: 1,234

--------------------------------------------------

Welcome to this tutorial on machine learning fundamentals. In this video, we will discuss the key concepts you need to know. Let's start with supervised learning algorithms.
```

### File Details

- **Format**: Plain text (.txt)
- **Encoding**: UTF-8
- **Line Endings**: Windows (CRLF)
- **Metadata**: Includes generation timestamp, language, model used, and word count
- **Timestamps**: MM:SS format for easy reference

---

## Configuration

### Settings Storage

The application automatically saves your preferences:

- **Location**: `%APPDATA%\youtube_whisper_transcriber\settings.json`
- **Managed Automatically**: No manual editing required

### Saved Settings

- Output directory path
- Last selected Whisper model
- Timestamp format preference
- Proxy configuration (if enabled)
- Window size and position

### Whisper Model Cache

Downloaded models are cached for faster subsequent use:

- **Location**: `%USERPROFILE%\.cache\whisper`
- **Persistence**: Models remain cached across application restarts
- **Storage**: Varies by model size (39MB to 1550MB per model)

### Environment Variables (Optional)

Advanced users can override settings using environment variables:

```bash
DEFAULT_WHISPER_MODEL=base
WHISPER_DEVICE=cuda
DEFAULT_OUTPUT_DIRECTORY=C:\Transcriptions
ENABLE_PROXY=true
PROXY_HOST=127.0.0.1
PROXY_PORT=8080
```

---

## Troubleshooting

### Common Issues

#### "FFmpeg not found" error

**Cause**: FFmpeg is not installed or not in system PATH

**Solution**:
- For distributed executable: FFmpeg is bundled—this error should not occur
- For source installation: Download FFmpeg from [ffmpeg.org](https://ffmpeg.org) and add to PATH

#### Download fails or network errors

**Cause**: Internet connection issues, geographic restrictions, or platform blocking

**Solutions**:
1. Check your internet connection
2. Try enabling Proxy/VPN settings (Settings → Proxy/VPN Settings)
3. Verify the video URL is accessible in your browser
4. For Instagram: Ensure the post is public

#### Model download fails

**Cause**: Network issues, insufficient disk space, or firewall blocking

**Solutions**:
1. Ensure stable internet connection
2. Free up disk space (models range from 39MB to 1550MB)
3. Check firewall settings
4. If using proxy, ensure proxy configuration is correct

#### Transcription accuracy issues

**Cause**: Using a smaller model or poor audio quality

**Solutions**:
1. Try a larger Whisper model (medium or large)
2. Note: Larger models require more RAM and take longer to process
3. Ensure source video has clear audio

#### "Out of memory" error

**Cause**: Insufficient RAM for selected model

**Solutions**:
1. Use a smaller Whisper model (tiny or base)
2. Close other applications to free up memory
3. Ensure your system meets minimum RAM requirements

#### Application won't start

**Cause**: Missing dependencies or Python version mismatch

**Solutions**:
1. For executable: Try running as administrator
2. For source: Verify Python 3.11+ is installed
3. For source: Reinstall dependencies: `pip install -r requirements.txt`

---

## Technical Information

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.11+ | Core application logic |
| **GUI Framework** | Tkinter | Desktop user interface |
| **ML Framework** | PyTorch | Whisper model execution |
| **Speech Recognition** | OpenAI Whisper | Audio transcription |
| **YouTube Download** | yt-dlp | YouTube video downloading |
| **Instagram Download** | instaloader | Instagram media downloading |
| **Audio Processing** | FFmpeg | Audio extraction and conversion |
| **HTTP Library** | requests | Network operations |
| **Proxy Support** | PySocks | SOCKS proxy handling |
| **Logging** | loguru | Application logging |
| **Packaging** | PyInstaller | Executable creation |

### Architecture

The application follows an **MVC pattern with workflow orchestration**:

```
├── src/
│   ├── main.py                  # Application entry point
│   ├── ui/                      # User Interface Layer
│   │   ├── main_window.py       # Main application window
│   │   ├── components.py        # Reusable UI components
│   │   └── proxy_settings_dialog.py  # Proxy configuration dialog
│   ├── controller/              # Workflow Orchestration
│   │   └── app_controller.py    # Transcription workflow coordinator
│   ├── processing/              # Processing Layer
│   │   ├── video_downloader.py  # YouTube video downloading
│   │   ├── instagram_downloader.py  # Instagram media downloading
│   │   └── whisper_transcriber.py   # Whisper transcription
│   ├── models/                  # Whisper model management
│   ├── utils/                   # Utilities
│   │   ├── state_manager.py     # Application state tracking
│   │   ├── error_handler.py     # Error handling and recovery
│   │   ├── proxy_manager.py     # Proxy configuration and rotation
│   │   └── validators.py        # URL and input validation
│   └── config/                  # Configuration
│       └── settings.py          # Settings management
```

### Key Dependencies

**Core ML and Audio:**
- `openai-whisper>=20231117` - Speech recognition
- `torch>=2.1.1` - PyTorch framework
- `torchaudio>=2.1.1` - Audio processing for PyTorch
- `transformers>=4.35.2` - Transformer models
- `numpy==1.25.2` - Numerical computing
- `scipy==1.11.4` - Scientific computing

**Media Download:**
- `yt-dlp>=2023.11.16` - YouTube video downloading
- `instaloader>=4.10.3` - Instagram media downloading
- `ffmpeg-python>=0.2.0` - FFmpeg Python wrapper

**Networking:**
- `requests>=2.31.0` - HTTP library
- `PySocks>=1.7.1` - SOCKS proxy support
- `urllib3>=2.1.0` - HTTP client
- `validators>=0.22.0` - URL validation

**System:**
- `psutil>=5.9.6` - System and process utilities
- `Pillow>=10.1.0` - Image processing
- `loguru>=0.7.2` - Logging framework

**Build:**
- `pyinstaller>=6.1.0` - Executable packaging

---

## Privacy and Legal

### Privacy Commitment

This application is designed with **privacy first**:

- **100% Local Processing** - All transcription occurs entirely on your computer
- **No Cloud Services** - No data is sent to external servers (except for downloading videos and models)
- **No Telemetry** - No usage tracking or analytics
- **No User Tracking** - No personal information is collected
- **Offline Transcription** - Once models are downloaded, transcription works offline

**What is downloaded:**
- Videos from YouTube/Instagram (temporary, deleted after transcription)
- Whisper models from OpenAI (one-time download, cached locally)

### Instagram Usage Warning

**IMPORTANT LEGAL NOTICE:**

Instagram functionality is provided for **educational and personal use only**. Downloading content from Instagram may violate Instagram's Terms of Service.

**User Responsibilities:**
- Users assume full legal responsibility for how they use this software
- Only download content you have explicit permission to access
- Respect copyright and intellectual property rights
- Comply with Instagram's Terms of Service
- Do not use for commercial purposes without proper authorization
- Do not redistribute downloaded content without permission

**The developers of this software:**
- Provide this functionality "as is" without warranties
- Are not responsible for misuse of the software
- Recommend using only on content you own or have permission to download

### License

This project is licensed under the **MIT License**.

**Copyright © 2024 OPBL Project**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Credits

This application is built using exceptional open-source technologies:

### Core Technologies

- **[OpenAI Whisper](https://github.com/openai/whisper)** - State-of-the-art speech recognition models
- **[PyTorch](https://pytorch.org/)** - Machine learning framework for model execution
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** - YouTube video downloading library
- **[Instaloader](https://github.com/instaloader/instaloader)** - Instagram media downloading library
- **[FFmpeg](https://ffmpeg.org/)** - Multimedia processing for audio extraction

### Development Tools

- **[Python](https://www.python.org/)** - Programming language
- **[Tkinter](https://docs.python.org/3/library/tkinter.html)** - GUI framework (included with Python)
- **[PyInstaller](https://www.pyinstaller.org/)** - Python application packaging

### Supporting Libraries

- **[loguru](https://github.com/Delgan/loguru)** - Elegant logging
- **[requests](https://requests.readthedocs.io/)** - HTTP library
- **[PySocks](https://github.com/Anorov/PySocks)** - SOCKS proxy support
- **[psutil](https://github.com/giampaolo/psutil)** - System utilities
- **[Pillow](https://python-pillow.org/)** - Image processing

---

## Support

### Getting Help

- **User Guide**: Press `F1` in the application or access Help → User Guide
- **Keyboard Shortcuts**: Help → Keyboard Shortcuts
- **GitHub Issues**: [Report bugs or request features](https://github.com/RiseDial/youtube-whisper-transcriber/issues)

### Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Made with ❤️ using OpenAI Whisper | 100% Local Processing | Privacy First**
