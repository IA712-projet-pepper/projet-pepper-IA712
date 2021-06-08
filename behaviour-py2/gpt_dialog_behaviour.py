#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: A Simple class to get & read FaceDetected Events"""

import qi
import time
import sys
import argparse

import os
from pathlib import Path


class GPTDialogBehaviour:
    """
    A simple class to react to face detection events.
    """

    def __init__(self, app):
        """
        Initialisation of qi framework and event detection.
        """

        app.start()
        session = app.session
        # Get the service ALMemory.
        self.tts = session.service("ALTextToSpeech")
        self.dialog_data_path = "../data/gpt_dialog"

    def run(self):
        """
        Loop on, wait for events until manual interruption.
        """
        print("Starting GPTDialog")

        try:
            while True:
                text_files = os.listdir(self.dialog_data_path)
                if len(text_files) == 0:
                    time.sleep(1)
                else:
                    with open(self.dialog_data_path + "/reponse.txt") as f:
                        text = f.readline().strip()
                        self.tts.say(text)
                        os.remove(self.dialog_data_path + "/reponse.txt")

        except KeyboardInterrupt:
            print "Interrupted by user, stopping GPTDialog"
            # stop
            sys.exit(0)


if __name__ == "__main__":

    IP = "192.168.2.147"
    PORT = 9559

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default=IP,
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=PORT,
                        help="Naoqi port number")

    args = parser.parse_args()
    try:
        # Initialize qi framework.
        connection_url = "tcp://" + args.ip + ":" + str(args.port)
        app = qi.Application(["Dialog GPT Behaviour", "--qi-url=" + connection_url])
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    human_dialog = GPTDialogBehaviour(app)
    human_dialog.run()
    #
    # print("hi")
    # session = qi.Session()
    # try:
    #     session.connect("tcp://" + args.ip + ":" + str(args.port))
    # except RuntimeError:
    #     print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
    #            "Please check your script arguments. Run with -h option for help.")
    #     sys.exit(1)
