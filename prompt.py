#!/usr/bin/env python3
import base64
from io import BytesIO
import json
import sys
from prompt2sand import Prompt2Sand
import os
import platform

def main():
    """Process a voice command, generate an image, and create a sand pattern."""
    if len(sys.argv) < 2:
        print("Usage: python prompt.py 'your drawing prompt'")
        sys.exit(1)
    
    prompt = sys.argv[1]
    p2s = Prompt2Sand()
    PATTERNS_DIR = "patterns"
    # Platform detection
    IS_RPI = platform.system() == "Linux" and os.path.exists('/proc/device-tree/model') and 'raspberry pi' in open('/proc/device-tree/model').read().lower()

    result = {
        "status": "failed",
        "message": "Failed to parse prompt",
        "image_base64": ""
    }

    stop_prompt = p2s.extract_stop_prompt(prompt)
    shutdown_prompt = p2s.extract_shutdown_prompt(prompt)
    restart_prompt = p2s.extract_restart_prompt(prompt)
    if stop_prompt:
        result["message"] = "Stopping DuneWeaver execution..."
        p2s.stop_execution()
    elif shutdown_prompt and IS_RPI:
        result["message"] = "Shutting down..."
        os.system("sudo shutdown now")
    elif restart_prompt and IS_RPI:
        result["message"] = "Restarting..."
        os.system("sudo reboot")
    else:
        # Extract drawing prompt if present
        draw_prompt = p2s.extract_draw_prompt(prompt)
        if draw_prompt:
            print(f"Extracted drawing prompt: {draw_prompt}")
            pattern_path = os.path.join(PATTERNS_DIR, f"{draw_prompt.replace(' ', '_')}.thr")
            theta_rho_file = os.path.join("custom_patterns", os.path.basename(pattern_path)).replace('\\', '/')
            theta_rho_files = p2s.list_theta_rho_files()
            # check our list of theta_rho files. If its none or we already have a match to the theta_rho_file, skip the image generation
            if theta_rho_files is None:
                print(f"No theta_rho files found")
                result["message"] = "Cannot reach DuneWeaver."
            elif any(theta_rho_file in file for file in theta_rho_files):
                print(f"Skipping image generation for: {draw_prompt} because it already exists")
                runResponse = p2s.run_theta_rho(theta_rho_file)
                if "success" in runResponse and runResponse["success"]:
                    result["status"] = "success"
                    result["message"] = f"Weaving existing dunes for: {draw_prompt}"
                else:
                    print(f"Error running theta_rho: {runResponse['detail']}")
                    result["message"] = f"Sorry, I couldn't weave the dunes. {runResponse['detail']}"
            else:
                # Process the draw_prompt
                image = p2s.generate_image_with_gemini(draw_prompt)

                if image:
                    # Convert the image to base64
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    result["image_base64"] = img_str
                    
                    # Convert to sand pattern
                    pattern = p2s.convert_image_to_sand(image)
                    
                    if pattern:
                        # Save pattern to file
                        with open(pattern_path, 'w') as f:
                            f.write(pattern['formatted_coords'])
                        print(f"Sand pattern saved to {pattern_path}")
                        
                        # Upload and run on DuneWeaver
                        uploadResponse = p2s.upload_theta_rho(pattern_path)
                        # check if response has a "success" key and if it's true
                        if "success" in uploadResponse and uploadResponse["success"]:
                            theta_rho_file = os.path.join("custom_patterns", os.path.basename(pattern_path)).replace('\\', '/')
                            runResponse = p2s.run_theta_rho(theta_rho_file)
                            if "success" in runResponse and runResponse["success"]:
                                result["status"] = "success"
                                result["message"] = f"Weaving the dunes for: {prompt}"
                            else:
                                result["message"] = f"Sorry, I couldn't weave the dunes. {runResponse['detail']}"
                        else:
                            result["message"] = f"Sorry, I couldn't upload the pattern to DuneWeaver. {uploadResponse['detail']}"
                    else:
                        result["message"] = "Failed to convert image to sand pattern"
                else:
                    result["message"] = "Failed to generate image"
        else:
            result["message"] = "Sorry, I can only weave dunes. Ask me to draw something."
    return result

if __name__ == "__main__":
    result = main()
    print(json.dumps(result)) 