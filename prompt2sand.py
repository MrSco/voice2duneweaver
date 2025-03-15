import requests
import re
from typing import Callable, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from image2sand import Image2Sand
import os
import cv2
import numpy as np


class Prompt2Sand:
    """
    A class to handle voice prompt to sand pattern conversion.
    It processes voice commands, generates images with Gemini, and interfaces with DuneWeaver.
    """
    
    def __init__(self):
        """Initialize the Prompt2Sand class with necessary API clients and settings."""
        # Load environment variables from .env file
        load_dotenv()
        
        # DuneWeaver settings
        self.dw_url = os.getenv('DW_URL')
        
        # Gemini API settings
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.genai_client = genai.Client(api_key=self.gemini_api_key)

        # Draw prompt
        self.draw_prompt = os.getenv('DRAW_PROMPT')
    
    def generate_image_with_gemini(self, prompt: str) -> Optional[Image.Image]:
        """Generate an image using Gemini 2.0 Flash Experimental"""
        try:
            prompt = self.draw_prompt.replace('{{}}', prompt)
            print(f"Generating image: {prompt}")
            response = self.genai_client.models.generate_content(
                model="models/gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
            )
            
            # Extract the image from the response
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    return Image.open(BytesIO(part.inline_data.data))
                elif part.text is not None:
                    print(f"Model response: {part.text}")
            
            return None
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
        
    def convert_image_to_sand(self, image: Image.Image) -> Optional[str]:
        # Convert image to sand pattern
        try:
            print("Converting image to sand pattern...")
            image2sand = Image2Sand()
            
            # Convert PIL Image to OpenCV format (numpy array)
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Configure options for sand pattern generation
            options = {
                'epsilon': 0.5,  # Controls point density
                'contour_mode': 'Tree',  # Use tree mode for better pattern detection
                'is_loop': True,  # Create continuous patterns
                'minimize_jumps': True,  # Optimize path to minimize jumps
                'output_format': 2,  # Use .thr format for Dune Weaver
                'max_points': 300  # Limit number of points for smooth operation
            }
            
            # Process the image directly and generate coordinates
            return image2sand.process_image(img_array, options)            
        except Exception as e:
            print(f"Error converting image to sand pattern: {e}")
            return None
    
    # handle prompt cases
    def handle_prompt_cases(self, text: str, callback: Callable = None, callback_cleanup: Callable = None, IS_RPI: bool = False):
        draw_prompt = None
        if self.extract_stop_prompt(text):
            if callback:
                callback("Stopping DuneWeaver execution...")
            self.stop_execution()
        elif self.extract_shutdown_prompt(text) and IS_RPI:
            if callback_cleanup:
                callback_cleanup("Shutting down...")
            os.system("sudo shutdown now")
        elif self.extract_restart_prompt(text) and IS_RPI:
            if callback_cleanup:
                callback_cleanup("Restarting...")
            os.system("sudo reboot")
        else:
            # Extract drawing prompt if present
            draw_prompt = self.extract_draw_prompt(text)
        return draw_prompt

    def extract_draw_prompt(self, text: str) -> Optional[str]:
        """Extract the drawing prompt from the transcribed text"""
        text = text.lower()
        prompt = None
        # using a regex, check for draw, create, make, generate, etc.
        # capture the keyword and everything after it
        match = re.search(r"\b(draw|create|make|generate)\b(.*)", text)
        if match:
            prompt = match.group(2).strip()  # Get the content after the command and strip whitespace
        else:
            prompt = text.strip()
            
        # Strip articles like "a", "an", "the" from the beginning of the prompt
        prompt = re.sub(r"^\s*(a|an|the)\s+", "", prompt)
        return prompt
    
    def extract_stop_prompt(self, text: str) -> bool:
        """Extract the stop prompt from the transcribed text"""
        text = text.lower()
        # using a regex, check for stop, exit, quit, etc.
        match = re.search(r"\b(stop|exit|quit)\b", text)
        if match:
            return True
        return False
    
    def extract_shutdown_prompt(self, text: str) -> bool:
        """Extract the shutdown prompt from the transcribed text"""
        text = text.lower()
        # using a regex, check for shutdown, power off, etc.
        match = re.search(r"\b(shutdown|shut down|power off|turn off)\b", text)
        if match:
            return True
        return False
    
    def extract_restart_prompt(self, text: str) -> bool:
        """Extract the restart prompt from the transcribed text"""
        text = text.lower()
        # using a regex, check for restart, reboot, etc.
        match = re.search(r"\b(restart|reboot)\b", text)
        if match:
            return True
        return False
    
    def list_theta_rho_files(self):
        """ Do a GET request to the theta_rho API with a 5 second timeout """
        url = f"{self.dw_url}/list_theta_rho_files"
        response = None
        try:
            response = requests.get(url, timeout=5).json()
        except Exception as e:
            print(f"Error listing theta_rho files: {e}")
        finally:
            return response
    
    def upload_theta_rho(self, pattern_path: str):
        """ Do a POST request to the theta_rho API """
        url = f"{self.dw_url}/upload_theta_rho"
        response = None
        with open(pattern_path, 'rb') as f:
            files = {'file': f}
            try:
                response = requests.post(url, files=files, timeout=5).json()
            except Exception as e:
                print(f"Error uploading theta_rho: {e}")
                response = {"detail": str(e)}
            finally:
                return response
    
    def run_theta_rho(self, pattern_path: str):
        """ Do a POST request to the theta_rho API """
        url = f"{self.dw_url}/run_theta_rho"
        response = None
        data = {
            'file_name': pattern_path,
            'pre_execution': 'adaptive'
        }
        print(f"Running theta_rho with data: {data}")
        try:
            response = requests.post(url, json=data, timeout=5).json()
            #print(response.json())
        except Exception as e:
            print(f"Error running theta_rho: {e}")
            response = {"detail": str(e)}
        finally:
            return response
    
    def stop_execution(self):
        url = f"{self.dw_url}/stop_execution"
        print(f"Stopping DuneWeaver execution...")
        response = None
        try:
            response = requests.post(url, timeout=5).json()
        except Exception as e:
            print(f"Error stopping DuneWeaver execution: {e}")
            response = {"detail": str(e)}
        finally:
            return response