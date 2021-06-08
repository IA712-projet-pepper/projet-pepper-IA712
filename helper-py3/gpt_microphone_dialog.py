# Librairies
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import speech_recognition as sr
import os
import time

# Load Dialogue GPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Initiate microphone and recognizer
mic = sr.Microphone()
r = sr.Recognizer()
print("Models loaded.")

path = "C:/Users/joach/Downloads/Pepper/"
count = 0
chat_history_ids = torch.tensor([])

# Calibrating background noise
print("- DialoGPT: Please wait. Calibrating microphone...")
with mic as source:
    r.adjust_for_ambient_noise(source, duration=5)
    
# Loop for dialogue
while True:
    
    # If no response is waiting
    if len(os.listdir(path)) == 0:
        
        # Get the audio
        print("- DialoGPT: Say something!")
        with mic as source:
            audio = r.listen(source, phrase_time_limit = 10, timeout = 10)
            print('.')
        
        
        try: # If speech recognizion works 
            print('.')
            text = r.recognize_google(audio, language = 'en-US')
        except: # If speech recognizion doens't work
            print("- DialoGPT: I did not understand...")
            text = "I did not understand"
            
            # Export the result 
            with open('response.txt', mode ='w') as file:
                file.write(response)

            time.sleep(0.3)
            continue
        
        # Generate an answer
        print("- User: {}".format(text))
        new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids[:,-10:], new_user_input_ids], dim=-1) if chat_history_ids.shape[0] > 0 else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print("- DialoGPT: {}".format(response))

        # Export the result 
        with open(path + 'response.txt', mode ='w') as file:
            file.write(response)
        
        # Wait a little bit
        time.sleep(0.3)
