from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import speech_recognition as sr
import os

path = "C:/Users/joach/Downloads/Pepper"
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

mic = sr.Microphone()
r = sr.Recognizer()
print("Models loaded.")

count = 0
chat_history_ids = torch.tensor([])

print("- DialoGPT: Please wait. Calibrating microphone...")
with mic as source:
    r.adjust_for_ambient_noise(source, duration=5)

if len(os.listdir(path)) == 0:

    print("- DialoGPT: Say something!")
    with mic as source:
        audio = r.listen(source, phrase_time_limit = 10, timeout = 10)
        print('.')
    try:
        print('.')
        text = r.recognize_google(audio, language = 'en-US')
    except:
        print("- DialoGPT: I did not understand...")
        response = "I did not understand"
        # export the result 
        with open('response.txt', mode ='w') as file:
            file.write(response)
        return

    print("- User: {}".format(text))
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids[:,-10:], new_user_input_ids], dim=-1) if chat_history_ids.shape[0] > 0 else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("- DialoGPT: {}".format(response))

    # export the result 
    with open('response.txt', mode ='w') as file:
        file.write(response)