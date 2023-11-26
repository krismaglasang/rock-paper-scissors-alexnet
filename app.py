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
    gr.Markdown("<p style='text-align:center; font-size:24px; font-weight:bold;'>Image Classification using Transfer Learning</p>")
    gr.Markdown("<p style='text-align:left; font-size:15px;'>Transfer learning using AlexNet trained on the rock-paper-scissors dataset curated by Laurence Moroney.</p>")
    gr.Markdown("<p style='text-align:left; font-size:15px;'>Note: This model sucks. It was trained only against 3D generated images, and can't even get some of those right. Also, \
                if you want to test it out, make sure to only use PIL or PNG image formats. Gradio (or at least I think the issue is with Gradio) craps out if you feed it JPEG.</p>")
    gr.Interface(
        fn=inference_fn,
        inputs=gr.Image(type='pil'),
        outputs='text',
        allow_flagging='never'
    )
        
demo.launch()