from typing import Literal
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms
from torch import Tensor
from torch.nn import MSELoss
from torch.nn import MaxPool2d, AvgPool2d
from torch import randn
from torch.optim import Adam, LBFGS
from torch import mm
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torch

#VGG16's architecture: 2*conv pool[1] 0  1  2  3  4 -> conv1_1, conv1_2
#                      2*conv pool[2] 5  6  7  8  9 -> conv2_1, conv2_2
#                      3*conv pool[3] 10 11 12 13 14 15 16 -> conv3_1, conv3_2, conv3_3
#                      3*conv pool[4] 17 18 19 20 21 22 23 -> conv4_1, conv4_2, conv4_3
#                      3*conv pool[5] 24 25 26 27 28 29 30 -> conv5_1, conv5_2, conv5_3

# so we we want the output of the first conv layer of each stage for [style reconstruction]
# as well as second conv in stage 4 for [content reconstruction]



class StyleTransferer:
    def __init__(self, model : Literal["VGG16", "VGG19", "AlexNet"] , pooling_type : Literal["MaxPooling", "AVGPooling"]):
        self.model_name = model
        self.pooling_type = pooling_type
        self.model_layers = {}
        self._load_model()
        self._setup_model()
        self._set_model_layers()
        # self.model = vgg16(pretrained = True).features

        # print(self.model)



    ## PRIVATE_METHODS
    def _load_model(self) -> None:
        self.model = vgg16(weights = VGG16_Weights).features
        

    def _setup_model(self) -> None:
        def freeze_weights():
            for param in self.model.parameters():
                param.requires_grad = False


        if self.pooling_type == "AVGPooling":
            for (i, (name, layer)) in enumerate(self.model._modules.items()):
                if isinstance(layer, MaxPool2d):
                    self.model[i] = AvgPool2d(2, 2)

        freeze_weights()


        self.model.eval()


    def _pass_image_through_network(self, tensor : Tensor) ->Tensor:
        fmaps = {}
        for name, layer in self.model.named_children():
            tensor = layer(tensor)
            if int(name) in self.layers_indices:
                fmaps[int(name)] = tensor
        return fmaps
            

    def _compute_gram_matrix(self, feature_maps : Tensor) -> Tensor:
        B, C, W, H = feature_maps.size()
        feature_maps = feature_maps.view(C, W*H)
        # print(type(feature_maps))
        # gram_matrix = feature_maps.mm(feature_maps.T)
        gram_matrix = mm(feature_maps, feature_maps.T)
        gram_matrix/=(C*W*H)

        return gram_matrix


    def _load_image(self, path: str):
        pass
    
    def _set_model_layers(self):
        match self.model_name:
            case 'VGG16': 
                self.layers_indices = [0, 5, 10, 17, 18, 24]  
                self.reconstruction_layer_idx = 18
                self.layers_weights = {0:1.0, 5:0.8, 10:0.6, 17:0.4, 24:0.2}
            case 'VGG19': pass
            case 'AlexNet': pass
        

    def save_vgg_tensor_as_jpeg(self, tensor, path):
        # undoing the transformations
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        mean = torch.tensor(
            [123.68, 116.779, 103.939],
            device=tensor.device
        ).view(3, 1, 1)

        with torch.no_grad():
            img = tensor + mean
            img = img.clamp(0, 255)
            img = img / 255.0

            to_pil_image(img).save(path, format="JPEG", quality=95)


    def transfer_image_style(self, content_image : Tensor = None, style_image : Tensor = None, content_image_path : str = None, style_image_path : str = None, start_with_content = True, cross_compute_gram_matrix = False, epochs: int = 100, alpha : float = 1, beta : float = 1e5, intialize_with_noise = True, learning_rate = 0.001) -> Tensor:
        self.alpha = alpha
        self.beta = beta

        content_image = Image.open(content_image_path).convert('RGB')
        style_image = Image.open(style_image_path).convert('RGB')

        NEW_SIZE = 400
        trans = transforms.Compose([
            transforms.Resize(NEW_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255),
            transforms.Normalize(
                mean=[123.68, 116.779, 103.939],
                std=[1, 1, 1]
            )
        ])



        content_image = trans(content_image)
        style_image = trans(style_image)

        content_image = content_image.unsqueeze(0)
        style_image = style_image.unsqueeze(0)


        content_image_fmaps = self._pass_image_through_network(content_image)
        style_image_fmaps = self._pass_image_through_network(style_image)

        style_image_gram_matrices = {}
        for i in self.layers_indices:
            style_image_gram_matrices[i] = self._compute_gram_matrix(style_image_fmaps[i])





        # target_image = randn(1, 3, 224, 224, requires_grad=True)
        target_image = content_image.clone().requires_grad_(True)
        loss_fn = MSELoss()
        optim = LBFGS([target_image], max_iter=20)
        for i in range(epochs):
            print(i)
            def closure():
                optim.zero_grad()

                target_image_fmaps = self._pass_image_through_network(target_image)

                content_loss = loss_fn(
                    target_image_fmaps[self.reconstruction_layer_idx],
                    content_image_fmaps[self.reconstruction_layer_idx]
                )

                style_loss = 0
                for j in self.layers_weights:
                    target_gram = self._compute_gram_matrix(target_image_fmaps[j])
                    style_loss += self.layers_weights[j] * loss_fn(
                        target_gram, style_image_gram_matrices[j]
                    )

                total_loss = self.beta * style_loss + self.alpha * content_loss
                total_loss.backward()
                return total_loss

            optim.step(closure)
            
            self.save_vgg_tensor_as_jpeg(target_image, f"outputs/{i}__")

        return target_image





    ## SETTERS
    def set_alpha(self, alpha : float) -> None:
        self.alpha = alpha
    
    def set_beta(self, beta : float) -> None:
        self.beta = beta

    def set_model(self, model : Literal["VGG16", "VGG19", "AlexNet"]) -> None:
        pass

    def set_pooling_type(self, pooling_type : Literal["MaxPooling", "AVGPooling"]) -> None:
        self.pooling_type = pooling_type
        ## TODO: complete
        pass

    def set_initalize_with_noise(self, state = bool):
        pass

    ## review all
