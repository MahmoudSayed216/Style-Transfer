from StyleTransfer import StyleTransferer
import torch
from torchvision.transforms.functional import to_pil_image
import sys





if __name__ == "__main__":
    args = sys.argv    
    

    content_img_path = args[1]
    style_img_path = args[2]
    output_path = args[3]
    new_size = int(args[4])
    
    obj = StyleTransferer('VGG16', pooling_type='MaxPooling', output_path=output_path, device='cuda', new_size = new_size)

    image = obj.transfer_image_style(content_image_path=content_img_path, style_image_path=style_img_path, alpha=1, beta=1_000_000, epochs=40, learning_rate=0.001)


