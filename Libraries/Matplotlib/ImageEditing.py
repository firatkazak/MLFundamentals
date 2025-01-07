import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Resmi yükle
input_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Inputs/batman.jpg"
output_path = "C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/Matplotlib/Data/Outputs/batman_edited.jpg"
image = Image.open(input_path)

# Şekil oluştur ve resmi göster
fig, ax = plt.subplots(figsize=(8, 6))

# Gri tonlama için resmi dönüştür
image_gray = image.convert("L")
ax.imshow(image_gray, cmap="gray")

# Çerçeve ekleme
rect = patches.Rectangle(
    (50, 50),  # Sol üst köşe
    image.size[0] - 100,  # Genişlik
    image.size[1] - 100,  # Yükseklik
    linewidth=5,
    edgecolor="red",
    facecolor="none",
)
ax.add_patch(rect)

# Metin ekleme
ax.text(
    x=10,
    y=10,
    s="Batman & Robin",
    fontsize=20,
    color="yellow",
    weight="bold",
    backgroundcolor="black",
)

# Eksenleri gizle ve resmi kaydet
ax.axis("off")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Resim başarıyla düzenlendi ve kaydedildi: {output_path}")
