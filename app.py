import gradio as gr
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import alexnet

class_labels = ['Paper', 'Rock', 'Scissors']

alex = alexnet(pretrained=True)
alex.classifier[6] = nn.Sequential(nn.Linear(
    in_features=4096,
    out_features=3
))
alex.load_state_dict(torch.load('./model.pth'))

print(f'Alex model: {alex}')

def inference_fn(raw_img):
    alex.eval()
    composer = Compose([Resize(224), ToTensor()])
    img = composer(raw_img)
    # Unsqueeze to add the batch dimension
    with torch.no_grad():
        output = alex(img.unsqueeze(0))
    print(f'output: {output}')
    pred_idx = torch.argmax(output, dim=1).item()
    prediction = class_labels[pred_idx]
    return prediction

with gr.Blocks() as demo:
    gr.Interface(
        fn=inference_fn,
        inputs=gr.Image(type='pil'),
        outputs='text',
        allow_flagging='never'
    )
        
demo.launch()