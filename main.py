from StyleTransfer import StyleTransferer
import torch
from torchvision.transforms.functional import to_pil_image
import sys





if __name__ == "__main__":
    args = sys.argv    
    

    content_img_path = args[1]
    style_img_path = args[2]
    output_path = args[3]
    
    obj = StyleTransferer('VGG16', pooling_type='AVGPooling', output_path=output_path, device='cuda', new_size = 1000)

    image = obj.transfer_image_style(content_image_path=content_img_path, style_image_path=style_img_path, alpha=1, beta=100_000, epochs=20, learning_rate=0.001)




# def save_vgg_tensor_as_jpeg(tensor, path):
    
#     if tensor.dim() == 4:
#         tensor = tensor.squeeze(0)


#     #undoing the preprocessing 
#     mean = torch.tensor(
#         [123.68, 116.779, 103.939],
#         device=tensor.device
#     ).view(3, 1, 1)

#     with torch.no_grad():
#         img = tensor + mean
#         img = img.clamp(0, 255)
#         img = img / 255.0

#         to_pil_image(img).save(path, format="JPEG", quality=95)


# save_vgg_tensor_as_jpeg(image, "output1.jpeg")