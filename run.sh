#!/bin/bash
# Voice2DuneWeaver Linux/Raspberry Pi Run Script

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Voice2DuneWeaver application...${NC}"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Virtual environment not found!${NC}"
    echo -e "${YELLOW}Please run ./setup.sh first to set up the environment.${NC}"
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}.env file not found!${NC}"
    echo -e "${YELLOW}Please run ./setup.sh first to create an .env file and add your API keys.${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment.${NC}"
    echo -e "${YELLOW}Try running ./setup.sh again or manually activate with 'source .venv/bin/activate'${NC}"
    exit 1
fi

# Check for app.py
if [ ! -f "app.py" ]; then
    echo -e "${RED}app.py not found! Make sure you're in the correct directory.${NC}"
    exit 1
fi

# Detect Raspberry Pi
IS_RPI=false
if [ -f /proc/device-tree/model ] && grep -q "raspberry pi" /proc/device-tree/model; then
    IS_RPI=true
    echo -e "${BLUE}Running on Raspberry Pi hardware.${NC}"
    
    # Check for GPIO access
    if ! groups | grep -q "gpio"; then
        echo -e "${YELLOW}Warning: Current user may not have proper GPIO access.${NC}"
        echo -e "${YELLOW}Consider adding your user to the gpio group:${NC}"
        echo -e "${YELLOW}    sudo usermod -a -G gpio $USER${NC}"
        echo -e "${YELLOW}You'll need to log out and back in for this to take effect.${NC}"
    fi
fi

# Trap Ctrl+C for clean shutdown
trap cleanup INT

cleanup() {
    echo -e "\n${YELLOW}Exiting Voice2DuneWeaver...${NC}"
    deactivate
    exit 0
}

# Run the application
echo -e "${GREEN}Launching Voice2DuneWeaver... Press Ctrl+C to exit.${NC}"
python app.py
if [ $? -ne 0 ]; then
    echo -e "${RED}An error occurred while running the application.${NC}"
    deactivate
    exit 1
fi

# Deactivate virtual environment when done
deactivate
exit 0 