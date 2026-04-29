from StyleTransfer import StyleTransferer
import torch
from torchvision.transforms.functional import to_pil_image
import sys





if __name__ == "__main__":
    args = sys.argv    
    
    image_format = args[1]
    content_img_path = args[2]
    style_img_path = args[3]
    output_path = args[4]
    new_size = int(args[5])
    epochs = int(args[6])
    print("Initializing StyleTransfer Object")
    obj = StyleTransferer('VGG16', pooling_type='MaxPooling', output_path=output_path, device='cuda', new_size = new_size)
    print("Object Initialized")
    print("Operating on the input images")
    image = obj.transfer_image_style(image_format=image_format, content_image_path=content_img_path, style_image_path=style_img_path, alpha=1, beta=1_000_000, epochs=epochs, learning_rate=0.001)
    print("Operation Completed")

