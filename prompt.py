#!/usr/bin/env python3
import sys
from prompt2sand import Prompt2Sand
import os

def main():
    """Process a voice command, generate an image, and create a sand pattern."""
    if len(sys.argv) < 2:
        print("Usage: python prompt.py 'your drawing prompt'")
        sys.exit(1)
    
    prompt = sys.argv[1]
    p2s = Prompt2Sand()
    PATTERNS_DIR = "patterns"

    print(f"Processing prompt: {prompt}")
    pattern_path = os.path.join(PATTERNS_DIR, f"{prompt.replace(' ', '_')}.thr")
    theta_rho_file = os.path.join("custom_patterns", os.path.basename(pattern_path)).replace('\\', '/')
    theta_rho_files = p2s.list_theta_rho_files()
    # check our list of theta_rho files. If its none or we already have a match to the theta_rho_file, skip the image generation
    if theta_rho_files is None:
        print(f"No theta_rho files found")
        return "Cannot reach DuneWeaver."
    elif any(theta_rho_file in file for file in theta_rho_files):
        print(f"Skipping image generation for: {prompt} because it already exists")
        runResponse = p2s.run_theta_rho(theta_rho_file)
        if "success" in runResponse and runResponse["success"]:
            return f"Weaving the dunes for: {prompt}"
        else:
            print(f"Error running theta_rho: {runResponse['detail']}")
            return f"Sorry, I couldn't weave the dunes. {runResponse['detail']}"
    else:
        
        # Process the prompt
        image = p2s.generate_image_with_gemini(prompt)
        
        if image:
            # Save temporary image for debugging (optional)
            #temp_img_path = "temp_image.png"
            #image.save(temp_img_path)
            #print(f"Generated image saved to {temp_img_path}")
            
            # Convert to sand pattern
            pattern = p2s.convert_image_to_sand(image)
            
            if pattern:
                # Save pattern to file
                pattern_path = f"{prompt.replace(' ', '_')}.thr"
                with open(pattern_path, 'w') as f:
                    f.write(pattern)
                print(f"Sand pattern saved to {pattern_path}")
                
                # Upload and run on DuneWeaver
                uploadResponse = p2s.upload_theta_rho(pattern_path)
                # check if response has a "success" key and if it's true
                if "success" in uploadResponse and uploadResponse["success"]:
                    theta_rho_file = os.path.join("custom_patterns", os.path.basename(pattern_path)).replace('\\', '/')
                    runResponse = p2s.run_theta_rho(theta_rho_file)
                    if "success" in runResponse and runResponse["success"]:
                        return f"Weaving the dunes for: {prompt}"
                    else:
                        print(f"Error running theta_rho: {runResponse['detail']}")
                        return f"Sorry, I couldn't weave the dunes. {runResponse['detail']}"
                else:
                    print(f"Error uploading theta_rho: {uploadResponse['detail']}")
                    return f"Sorry, I couldn't upload the pattern to DuneWeaver. {uploadResponse['detail']}"
            else:
                return "Failed to convert image to sand pattern"
        else:
            return "Failed to generate image"

if __name__ == "__main__":
    result = main()
    print(result) 