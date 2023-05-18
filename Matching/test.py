from PIL import Image, ImageDraw, ImageFont



def draw_comparison_image(CO2_score):
    factor = 206
    flights = np.floor(CO2_score/factor)
    # Load the PNG image
    image = Image.open(r"./Local_files/airplane.jpeg")
    # Create a drawing context
    draw = ImageDraw.Draw(image, mode = "RGB")
    draw.fontmode = "L"
    image_width, image_height = image.size
    # Define the font and size you want to use
    font = ImageFont.truetype("times.ttf", 200)
    # Define the text you want to add
    text1 = "The CO2 equivalents corresponds to 2060"
    text2 = "round-trip flights between Oslo and Trondheim"
    text_bbox1 = draw.textbbox((0, 0), text1, font=font)
    text_bbox2 = draw.textbbox((0, 0), text2, font=font)

    # Calculate the width and height of the text
    text_width1 = text_bbox1[2] - text_bbox1[0]
    text_height1 = text_bbox1[3] - text_bbox1[1]
    text_width2 = text_bbox2[2] - text_bbox2[0]
    text_height2 = text_bbox2[3] - text_bbox2[1]

    # Calculate the position to place the text in the middle
    position1 = ((image_width - text_width1) // 2, 150 + text_height1)
    position2 = ((image_width - text_width2) // 2, 150 + text_height2*2 + 50)

    # Add the text to the image
    draw.text(position1, text1, font=font, fill=(256, 256, 256))
    draw.text(position2, text2, font=font, fill=(256, 256, 256))

    # Save the modified image
    image.save(r"./Local_files/comparison_image.png", dpi = (300, 300))