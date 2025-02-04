from googletrans import Translator
import requests
import json
import pygame
import time
import sys
import numpy as np
import soundfile as sf
import scipy.fftpack
import math
import random

# OpenGL Imports
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
# Inisialisasi translator dan pygame
translator = Translator()
pygame.mixer.init()

def chat_with_ai(messages):
    """Mengirim pesan ke model Llama menggunakan API Chat"""
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.2",  # Ganti dengan model yang Anda gunakan
        "messages": messages,
        "stream": False  # Nonaktifkan streaming untuk respons langsung
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return None

def save_message(role, content):
    try:
        with open('chat_history.json', 'r', encoding='utf-8') as f:
            chat_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        chat_history = []

    chat_history.append({"role": role, "content": content})

    with open('chat_history.json', 'w', encoding='utf-8') as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)

def get_chat_history():
    try:
        with open('chat_history.json', 'r', encoding='utf-8') as f:
            chat_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        chat_history = []

    return chat_history

def text_to_speech(text, speaker_id=29):
    """Mengkonversi teks ke suara menggunakan VoiceVox"""
    query_url = "http://localhost:50021/audio_query"
    params = {"text": text, "speaker": speaker_id}

    try:
        query_response = requests.post(query_url, params=params)
        query_response.raise_for_status()

        synthesis_url = "http://localhost:50021/synthesis"
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            synthesis_url,
            headers=headers,
            params={"speaker": speaker_id},
            data=json.dumps(query_response.json())
        )
        response.raise_for_status()

        pygame.mixer.music.stop()
        pygame.mixer.music.unload()

        with open("output.wav", "wb") as f:
            f.write(response.content)

        return True
    except Exception as e:
        print(f"Error in VoiceVox: {e}")
        return False

def play_audio(file_path):
    """Memainkan file audio"""
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
    except pygame.error as e:
        print(f"Error playing audio: {e}")

def main():
    print("Selamat datang! Ketik 'exit' untuk keluar")

    chat_history = get_chat_history()

    while True:
        user_input = input("\nAnda: ")

        if user_input.lower() == "exit":
            print("Keluar dari program...")
            pygame.quit()
            sys.exit()

        chat_history.append({"role": "user", "content": user_input})
        # save_message("user", user_input)

        ai_response = chat_with_ai(chat_history)
        if not ai_response:
            continue

        chat_history.append({"role": "assistant", "content": ai_response})
        # save_message("assistant", ai_response)

        print(f"\nAI: {ai_response}")
        translated = translator.translate(ai_response, dest='ja').text
        print(f"Terjemahan: {translated}")

        if text_to_speech(translated):
            pygame.init()
            WIDTH, HEIGHT = 800, 600
            pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
            pygame.display.set_caption("OpenGL Audio Visualizer")
            icon = pygame.image.load("logo38.png")
            pygame.display.set_icon(icon)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POINT_SMOOTH)
            glPointSize(5)
            glClearColor(0, 0, 0, 1)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, WIDTH / HEIGHT, 0.1, 50.0)
            glMatrixMode(GL_MODELVIEW)

            pygame.mixer.init()
            audio_file = "output.wav"
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            data, sample_rate = sf.read(audio_file)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            def get_audio_power(audio_chunk):
                fft_data = scipy.fftpack.fft(audio_chunk)
                power = np.abs(fft_data[:len(fft_data) // 2])
                return np.mean(power)

            running = True
            frame_index = 0
            FPS = 60
            clock = pygame.time.Clock()
            samples_per_frame = sample_rate // FPS

            def draw_visualizer(scale_factor):
                """Menggambar efek visual OpenGL"""
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glLoadIdentity()
                glTranslatef(0.0, 0.0, -3.0)  # Perbaiki posisi kamera

                glBegin(GL_POINTS)
                for _ in range(2000):  # Jumlah titik
                    angle = random.uniform(0, 2 * math.pi)
                    radius = random.uniform(0.1, 0.5) * scale_factor
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = random.uniform(-0.5, 0.5)  # Perbaiki rentang z

                    glColor3f(0.0, 0.0, 1.0)  # Warna putih untuk kontras
                    glVertex3f(x, y, z)
                glEnd()

                pygame.display.flip()
            while running:
                start = frame_index * samples_per_frame
                end = start + samples_per_frame
                if end >= len(data):
                    break

                audio_chunk = data[start:end]
                scale_factor = get_audio_power(audio_chunk)
                scale_factor = np.interp(scale_factor, (0, np.max(data)), (0.1, 2.0))

                draw_visualizer(scale_factor)



                frame_index += 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                clock.tick(FPS)

if __name__ == "__main__":
    main()