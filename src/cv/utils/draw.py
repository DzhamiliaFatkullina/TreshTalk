from PIL import Image, ImageDraw, ImageFont

def draw_boxes(image_path, detections, classifier_results):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for det, clf in zip(detections, classifier_results):

        x1, y1, x2, y2 = det["bbox"]
        label = f"{clf[0]['label']} ({clf[0]['confidence']:.2f})"

        draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
        draw.text((x1, y1 - 20), label, fill="red", font=font)

    return img
