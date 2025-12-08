from StyleTransfer import StyleTransferer
import torch
from torchvision.transforms.functional import to_pil_image


obj = StyleTransferer('VGG16', pooling_type='MaxPooling')

image = obj.transfer_image_style(content_image_path='src.png', style_image_path='dst.jpg', alpha=1, beta=100_000, epochs=80, learning_rate=0.001)


def save_vgg_tensor_as_jpeg(tensor, path):
    
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)


    #undoing the preprocessing 
    mean = torch.tensor(
        [123.68, 116.779, 103.939],
        device=tensor.device
    ).view(3, 1, 1)

    with torch.no_grad():
        img = tensor + mean
        img = img.clamp(0, 255)
        img = img / 255.0

        to_pil_image(img).save(path, format="JPEG", quality=95)


save_vgg_tensor_as_jpeg(image, "output1.jpeg")