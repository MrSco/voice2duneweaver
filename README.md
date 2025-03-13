# Voice2DuneWeaver

Voice2DuneWeaver is a voice-controlled application that generates sand art patterns using Google's Gemini AI. It allows users to create sand art designs by speaking simple voice commands, which are then processed and sent to a DuneWeaver sand table.

## Features

- 🗣️ **Voice Recognition**: Speak to create sand art designs
- 🤖 **AI-Powered**: Uses Google Gemini API to generate creative patterns
- 🏝️ **Sand Art Generation**: Converts AI-generated images into sand table patterns
- 🔌 **Cross-Platform**: Works on Windows, Linux, and Raspberry Pi
- 💡 **LED Support**: Visual feedback via ReSpeaker 2-Mic Hat on Raspberry Pi

## Requirements

### Hardware
- Microphone for voice input
- Speakers for voice feedback
- For Raspberry Pi:
  - Raspberry Pi (3B+ or later recommended)
  - ReSpeaker 2-Mic Hat (optional, for LED feedback)
  - Compatible sand table running DuneWeaver

### Software
- Python 3.7 or higher
- Google Gemini API key
- DuneWeaver running on your network

## Installation

### Setting up the Environment

1. Clone this repository:
   ```
   git clone https://github.com/mrsco/voice2duneweaver.git
   cd voice2duneweaver
   ```

2. Create a .env file with your API keys based on the .env.example template:
   ```
   # DuneWeaver settings
   DW_URL=http://localhost:8080
   
   # GOOGLE GEMINI API KEY
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. Use the provided setup scripts:
   - Windows: Run `setup.bat`
   - Linux/Raspberry Pi: Run `./setup.sh`

### Manual Setup (Alternative)

1. Create a virtual environment:
   ```
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Raspberry Pi: `source .venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the application using the run script:
   - Windows: Run `run.bat`
   - Linux/Raspberry Pi: Run `./run.sh`

2. Once the application starts, you can use the following voice commands:
   - "Draw a [description]" - Generates and sends a sand pattern
   - "Stop" - Cancels the current operation
   - "Exit" or "Quit" - Exits the application

3. Examples:
   - "Draw a spiral galaxy"
   - "Draw a geometric pattern with waves"
   - "Draw a minimalist landscape"

## Running as a Service

If you want Voice2DuneWeaver to start automatically on boot and run in the background, you can set it up as a service.

### Linux/Raspberry Pi

1. Create a systemd service file:
   ```
   sudo nano /etc/systemd/system/voice2duneweaver.service
   ```

2. Add the following content:
   ```
   [Unit]
   Description=Voice2DuneWeaver Service
   After=network.target

   [Service]
   Type=simple
   User=YOUR_USERNAME
   WorkingDirectory=/path/to/your/voice2duneweaver
   ExecStart=/bin/bash -c 'source /path/to/your/voice2duneweaver/.venv/bin/activate && python /path/to/your/voice2duneweaver/app.py'
   Restart=on-failure
   RestartSec=5s

   [Install]
   WantedBy=multi-user.target
   ```

3. Replace `YOUR_USERNAME` and `/path/to/your/voice2duneweaver` with your actual values

4. Enable and start the service:
   ```
   sudo systemctl daemon-reload
   sudo systemctl enable voice2duneweaver.service
   sudo systemctl start voice2duneweaver.service
   ```

5. Check service status:
   ```
   sudo systemctl status voice2duneweaver.service
   ```

## Raspberry Pi Specific Notes

When running on Raspberry Pi with the ReSpeaker 2-Mic Hat:
- The button on the hat can be used to start listening
- LEDs provide visual feedback about the application state:
  - Blue: Listening for commands
  - Purple: Processing
  - Green: Success
  - Red: Error

## Troubleshooting

- **No sound input detected**: Check your microphone settings and ensure it's properly connected
- **API key errors**: Verify your Gemini API key is correctly set in the .env file
- **Connection errors**: Ensure DuneWeaver is running and accessible at the URL specified in your .env file
- **GPIO errors on Raspberry Pi**: Make sure you have the necessary permissions to access GPIO pins
- **Service not starting**: Check the service logs using `sudo journalctl -u voice2duneweaver.service` or Event Viewer on Windows

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses Google's Gemini AI for image generation
- Special thanks to the DuneWeaver project (https://github.com/tuanchris/dune-weaver), and the Image2Sand project (https://github.com/orionwc/image2sand)
