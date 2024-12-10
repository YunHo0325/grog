import gradio as gr
from urllib.parse import urlparse
import requests
import time
import os

from utils.gradio_helpers import parse_outputs, process_outputs

inputs = []
inputs.append(gr.Image(
    label="Image", type="filepath"
))

inputs.append(gr.Textbox(
    label="Prompt", info=None
))

inputs.append(gr.Textbox(
    label="Negative Prompt", info='''Things you do not want in the image'''
))

inputs.append(gr.Number(
    label="Width", info=None, value=1024
))

inputs.append(gr.Number(
    label="Height", info=None, value=1024
))

inputs.append(gr.Number(
    label="Steps", info=None, value=20
))

inputs.append(gr.Number(
    label="Seed", info='''Fix the random seed for reproducibility''', value=None
))

inputs.append(gr.Number(
    label="Prompt Strength", info='''Strength of the prompt. This is the CFG scale, higher numbers lead to stronger prompt, lower numbers will keep more of a likeness to the original.''', value=7
))

inputs.append(gr.Number(
    label="Instant Id Strength", info='''How strong the InstantID will be.''', value=1
))

inputs.append(gr.Number(
    label="Ip Adapter Weight", info='''How much the IP adapter will influence the image''', value=0.2
))

inputs.append(gr.Number(
    label="Ip Adapter Noise", info='''How much noise is added to the IP adapter input''', value=0.5
))

inputs.append(gr.Checkbox(
    label="Upscale", info='''2x upscale the sticker''', value=False
))

inputs.append(gr.Number(
    label="Upscale Steps", info='''Number of steps to upscale''', value=10
))

names = ['image', 'prompt', 'negative_prompt', 'width', 'height', 'steps', 'seed', 'prompt_strength', 'instant_id_strength', 'ip_adapter_weight', 'ip_adapter_noise', 'upscale', 'upscale_steps']

outputs = []
outputs.append(gr.Image())
outputs.append(gr.Image())

expected_outputs = len(outputs)
def predict(request: gr.Request, *args, progress=gr.Progress(track_tqdm=True)):
    headers = {'Content-Type': 'application/json'}

    payload = {"input": {}}
    
    
    parsed_url = urlparse(str(request.url))
    base_url = parsed_url.scheme + "://" + parsed_url.netloc
    for i, key in enumerate(names):
        value = args[i]
        if value and (os.path.exists(str(value))):
            value = f"{base_url}/file=" + value
        if value is not None and value != "":
            payload["input"][key] = value

    response = requests.post("http://0.0.0.0:5000/predictions", headers=headers, json=payload)

    
    if response.status_code == 201:
        follow_up_url = response.json()["urls"]["get"]
        response = requests.get(follow_up_url, headers=headers)
        while response.json()["status"] != "succeeded":
            if response.json()["status"] == "failed":
                raise gr.Error("The submission failed!")
            response = requests.get(follow_up_url, headers=headers)
            time.sleep(1)
    if response.status_code == 200:
        json_response = response.json()
        #If the output component is JSON return the entire output response 
        if(outputs[0].get_config()["name"] == "json"):
            return json_response["output"]
        predict_outputs = parse_outputs(json_response["output"])
        processed_outputs = process_outputs(predict_outputs)
        difference_outputs = expected_outputs - len(processed_outputs)
        # If less outputs than expected, hide the extra ones
        if difference_outputs > 0:
            extra_outputs = [gr.update(visible=False)] * difference_outputs
            processed_outputs.extend(extra_outputs)
        # If more outputs than expected, cap the outputs to the expected number
        elif difference_outputs < 0:
            processed_outputs = processed_outputs[:difference_outputs]
        
        return tuple(processed_outputs) if len(processed_outputs) > 1 else processed_outputs[0]
    else:
        if(response.status_code == 409):
            raise gr.Error(f"Sorry, the Cog image is still processing. Try again in a bit.")
        raise gr.Error(f"The submission failed! Error: {response.status_code}")

title = "Demo for face-to-sticker cog image by fofr"
model_description = "Turn a face into a sticker"

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=model_description,
    allow_flagging="never",
)
app.launch(share=True)

