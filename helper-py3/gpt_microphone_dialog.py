# Librairies
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import speech_recognition as sr
import os
import time
import wikipedia

# Load Dialogue GPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Initiate microphone and recognizer
mic = sr.Microphone(2)
r = sr.Recognizer()
print("Models loaded.")

response_path = "../data/gpt_dialog"
response_file_path = response_path + "/reponse.txt"
chat_history_ids = torch.tensor([])
wiki_search = False

# Calibrating background noise
print("- DialoGPT: Please wait. Calibrating microphone...")
with mic as source:
    r.adjust_for_ambient_noise(source, duration=5)
    
# Loop for dialogue
while True:
    # If no response is waiting
    if len(os.listdir(response_path)) == 0:
        # Get the audio
        print("- DialoGPT: Say something!")
        with mic as source:
            audio = r.listen(source, phrase_time_limit = 10, timeout = 10)
            print('.')

        try:  # If speech recognizion works
            print('.')
            text = r.recognize_google(audio, language = 'en-US')
        except:  # If speech recognizion doens't work
            print("- DialoGPT: I did not understand...")
            response = "I did not understand. Can you repeat?"
            
            # Export the result 
            with open(response_file_path, mode ='w', encoding="utf-8") as file:
                file.write(response)

            time.sleep(0.3)
            continue
        
        print("- User: {}".format(text))
        
        if ("ok" in [word.lower() for word in text.split()]) and (len(text.split()) <= 3):  # Reset context
            chat_history_ids = torch.tensor([])
            bot_input_ids = torch.tensor([])
            print("- DialoGPT: .")
            continue
        
        elif wiki_search:  # Answer from wikipedia
            wiki_search = False
            search = wikipedia.search(text, results=1)
            if len(search) == 0:
                response = "I can't find anything about" + text + "on wikipedia."
            else:
                response = wikipedia.summary(search[0], auto_suggest = False, sentences = 2)
                
        elif ("wikipedia" in [word.lower() for word in text.split()]): # Wikipedia request
            response = "What do you want me to search on wikipedia?"
            wiki_search = True
            
        else : # Generate an answer with Dialogue GPT
            new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids.shape[0] > 0 else new_user_input_ids
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        print("- DialoGPT: {}".format(response))
        # Export the result 
        with open(response_file_path, mode ='w', encoding="utf-8") as file:
            file.write(response)
    else:
        # Wait a little bit
        time.sleep(0.3)
