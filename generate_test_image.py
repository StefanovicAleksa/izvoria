from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Ensure the resources directory exists
resources_dir = Path("integration_tests/resources")
resources_dir.mkdir(parents=True, exist_ok=True)

img_path = resources_dir / "test_vision_chart.jpg"

# Create a solid white background
img = Image.new('RGB', (800, 600), color='white')
draw = ImageDraw.Draw(img)

# Draw a fake "chart" (a blue rectangle)
draw.rectangle([200, 200, 400, 500], fill="blue")

# Add some highly specific text for the VLM to read
text = "Onyx Enterprise: Q3 Revenue reached $50M"
# We use default font since it's guaranteed to be installed
draw.text((100, 100), text, fill="black")

img.save(img_path, format="JPEG")
print(f"✅ Generated test image at: {img_path}")