from PIL import Image, ImageDraw, ImageFont

# Load the PNG image
image = Image.open(r"./Local_files/airplane.jpg")
# Create a drawing context
draw = ImageDraw.Draw(image, mode = "RGB")

# Define the font and size you want to use
font = ImageFont.truetype("times.ttf", 14)

# Define the position where you want to add the text
position1 = (60, 10)
position2 = (45, 25)

# Define the text you want to add
text1 = "The CO2 equivalents corresponds to 206"
text2 = "round-trip flights between Oslo and Trondheim"
# Add the text to the image
draw.text(position1, text1, font=font, fill=(256, 256, 256))
draw.text(position2, text2, font=font, fill=(256, 256, 256))

# Save the modified image
image.save(r"./Local_files/modified.png")

def draw_comparison_image():
    return 0