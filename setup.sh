#!/bin/bash
# Voice2DuneWeaver Linux/Raspberry Pi Setup Script

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Voice2DuneWeaver environment for Linux/Raspberry Pi...${NC}"

# Check for Python installation
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}Found $PYTHON_VERSION${NC}"
else
    echo -e "${RED}Python 3 not found! Please install Python 3.7 or later and try again.${NC}"
    exit 1
fi

# Detect Raspberry Pi
IS_RPI=false
if [ -f /proc/device-tree/model ] && grep -q "raspberry pi" /proc/device-tree/model; then
    IS_RPI=true
    echo -e "${BLUE}Detected Raspberry Pi hardware.${NC}"
    
    # Check for audio setup on Raspberry Pi
    if [ ! -f /etc/asound.conf ]; then
        echo -e "${YELLOW}Note: You may need to configure your audio device.${NC}"
        echo -e "${YELLOW}If using ReSpeaker 2-Mic Hat, follow the manufacturer's instructions.${NC}"
    fi
    
    # Check for GPIO access
    if ! groups | grep -q "gpio"; then
        echo -e "${YELLOW}Warning: Current user may not have proper GPIO access.${NC}"
        echo -e "${YELLOW}Consider adding your user to the gpio group:${NC}"
        echo -e "${YELLOW}    sudo usermod -a -G gpio $USER${NC}"
        echo -e "${YELLOW}You'll need to log out and back in for this to take effect.${NC}"
    fi
fi

# install apt packages
echo -e "${YELLOW}Installing apt packages...${NC}"
sudo apt install python3-dev libportaudio2 portaudio19-dev libffi-dev libssl-dev

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv .venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment.${NC}"
        echo -e "${YELLOW}You may need to install venv:${NC}"
        echo -e "${YELLOW}    sudo apt install python3-venv${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment.${NC}"
    exit 1
fi

# Check for pip in virtual environment
if ! python -m pip --version &>/dev/null; then
    echo -e "${RED}pip not found in virtual environment!${NC}"
    echo -e "${YELLOW}Attempting to install pip in the virtual environment...${NC}"
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    rm get-pip.py
    if ! python -m pip --version &>/dev/null; then
        echo -e "${RED}Failed to install pip in virtual environment.${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}Found pip in virtual environment.${NC}"

# Install requirements
echo -e "${YELLOW}Installing dependencies...${NC}"
python -m pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install dependencies.${NC}"
        echo -e "${YELLOW}If you're on Raspberry Pi, you might need:${NC}"
        echo -e "${YELLOW}    sudo apt install python3-dev libportaudio2 portaudio19-dev libffi-dev libssl-dev${NC}"
        exit 1
    fi
else
    echo -e "${RED}requirements.txt not found!${NC}"
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo -e "${YELLOW}Creating .env file from template...${NC}"
        cp .env.example .env
        echo -e "${YELLOW}Please edit the .env file to add your API keys.${NC}"
    else
        echo -e "${YELLOW}Creating basic .env file...${NC}"
        cat > .env << EOL
# DuneWeaver settings
DW_URL=http://localhost:8080

# GOOGLE GEMINI API KEY
GEMINI_API_KEY=
EOL
        echo -e "${YELLOW}Please edit the .env file to add your API keys.${NC}"
    fi
else
    echo -e "${GREEN}.env file already exists.${NC}"
fi

# Make run script executable
chmod +x run.sh

# Deactivate virtual environment
deactivate

echo -e "${GREEN}Setup complete! You can now run the application with ./run.sh${NC}" 