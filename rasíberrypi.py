from fastapi import FastAPI 
import requests as request
from gpiozero import Button
from signal import pause

server_host = "http://localhost:8000/api"

START_BUTTON = Button(3)
END_BUTTON = Button(4)

def start_game() -> None:    
    response = request.post(f"{server_host}/detection/start")
    print("Startando a visao computacional")
    return response

def end_game() -> None:
    response = request.post(f"{server_host}/detection/stop")
    print("Terminando a visao computacional")
    return response

def create_eletronic_logic() -> None:
    START_BUTTON.when_held = start_game 
    END_BUTTON.when_held = end_game

if __name__ == "__main__":
    create_eletronic_logic()
    pause()