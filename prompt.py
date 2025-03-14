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