import os
import json
from tkinter import Tk, Label
from PIL import Image, ImageTk
# Had to brew install python-tk for tk to work


def load_images(folder_path):
    """Load all image paths from the folder."""
    supported_formats = ['.png', '.jpg', '.jpeg']
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in supported_formats]

def display_image(image_path):
    """Display an image in a window, return True if displayed successfully, False if image is corrupted."""
    try:
        root = Tk()
        img = Image.open(image_path)
        img = img.resize((640, 480), Image.LANCZOS)  # Resize for convenience
        photo = ImageTk.PhotoImage(img)
        label = Label(root, image=photo)
        label.pack()
        root.mainloop()
        return True
    except Exception as e:
        print(f"Error displaying {image_path}: {e}")
        return False

def delete_image(image_path):
    """Delete the specified image file."""
    try:
        os.remove(image_path)
        print(f"Deleted {image_path}.")
    except Exception as e:
        print(f"Error deleting {image_path}: {e}")

def main(folder_path, output_file):
    images = load_images(folder_path)
    data = []
    
    for image in images:
        # Attempt to display the image
        print(f"Displaying {image}...")
        if not display_image(image):
            delete_image(image)
            continue
            
        
        # Get user input for the label
        output = input("Enter label ('yes' or 'no'), or 'd' to delete: ").strip().lower()
        if output == 'd':
            delete_image(image)
            continue

        while output not in ['yes', 'no']:
            print("Invalid input. Please enter 'positive' or 'negative'.")
            output = input("Enter label ('positive' or 'negative'): ").strip().lower()
        if output == 'd':
            delete_image(image)
            continue
        # Prepare the data entry
        entry = {
            "id": os.path.splitext(os.path.basename(image))[0],  # Assuming ID is the filename without extension
            "image": os.path.basename(image),
            "instruction": "Is this image offensive?",
            "output": output,
            "type": "detail"
        }
        data.append(entry)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print("All images have been labeled and saved to the JSON file.")

if __name__ == "__main__":
    images_folder = "/path/to/SGraw_Images"  # Update this to your images folder path
    output_json = "/path/to/labels.json"  # Output JSON filename
    main(images_folder, output_json)
