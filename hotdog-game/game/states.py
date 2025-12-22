import os
from abc import ABC, abstractmethod

import pygame
from game.config import *
from game.misc import Button

from time import time

import numpy as np

import game.edits as edits
import game.recognizer as recognizer

import cv2


class GameState(ABC):
    def __init__(self, game) -> None:
        self.game = game
        self.vm = game.vm

    def enter(self) -> None:
        pass

    @abstractmethod
    def handle_event(self, event: pygame.event.Event) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def draw(self) -> None:
        pass


class MainMenuState(GameState):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.title = "Hotdog Eater 3067"
        self.mascot = self.vm.load_image("durian.png", (200, 200))

        button_w, button_h = 500, 70
        x_center = (SCREEN_WIDTH - button_w) // 2
        y_center = (SCREEN_HEIGHT + 200 - button_h) // 2
        y_offset = -50

        self.buttons = [
            Button(
                self.vm,
                (x_center, y_center + y_offset),
                (button_w, button_h),
                "Start eating",
                on_click=self.start_eating,
            ),
            Button(
                self.vm,
                (x_center, y_center + y_offset + 5 + button_h),
                (button_w, button_h),
                "Help",
                on_click=self.help,
            ),
            Button(
                self.vm,
                (x_center, y_center + y_offset + (5 + button_h) * 2),
                (button_w, button_h),
                "Quit",
                on_click=self.quit_game,
            ),
        ]

    def start_eating(self) -> None:
        self.game.state = "eating"

    def help(self) -> None:
        self.game.state = "help"

    def quit_game(self) -> None:
        self.game.running = False

    def handle_event(self, event: pygame.event.Event) -> None:
        for btn in self.buttons:
            btn.handle_event(event)

    def update(self) -> None:
        pass

    def draw(self) -> None:
        self.vm.clear_screen(COLOR_BG_MENU)

        tw, th = self.vm.get_text_size(self.title, font_size=115)

        self.vm.draw_text(
            ((SCREEN_WIDTH - tw - 175) // 2, 175),
            self.title,
            COLOR_TEXT_TITLE,
            font_size=115,
        )

        self.vm.draw_image(((SCREEN_WIDTH + tw - 175) // 2, 125), self.mascot)
        for btn in self.buttons:
            btn.draw()


class EatingState(GameState):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.cap = None
        self.frame = None

        hotdog_image = cv2.imread(os.path.join("assets", "images", "hotdog.png"), cv2.IMREAD_UNCHANGED)
        self.hotdog_image = cv2.cvtColor(hotdog_image, cv2.COLOR_BGRA2RGBA)

        self.open_mouth_confidence = 0

        self.max_time = 15

        self._reset_states()

        self.state = "eating"  # "eating" or "interlude" or "results"
        self.player_id = 0

        self.results = []

    def _reset_states(self):
        self.cooldown = [False, False]
        self.score = 0
        self.start_time = -1
        self.timer = -1

    def enter(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 960)
        self.cap.set(4, 540)

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.state == "interlude" and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.state = "eating"
            elif event.key == pygame.K_RETURN:
                self.state = "results"
        elif self.state == "results" and event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.game.state = "menu"

    def update(self) -> None:
        if self.timer == 0:
            self.state = "interlude"
            self.player_id += 1

            self.results.append(self.score)

            self._reset_states()

        if self.state != "eating":
            return

        _, frame = self.cap.read()

        flipped = np.fliplr(frame)
        flipped_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        faces = self.game.face_detector.process(flipped_rgb).multi_face_landmarks
        hands = self.game.hands_detector.process(flipped_rgb)
        hands, hands_info = hands.multi_hand_landmarks, hands.multi_handedness

        if faces:
            face = faces[0]
            lm = face.landmark

            if recognizer.is_mouth_open(face):
                self.open_mouth_confidence = min(100, self.open_mouth_confidence + 5)
            else:
                self.open_mouth_confidence = max(0, self.open_mouth_confidence - 5)

            if self.open_mouth_confidence > 50:
                (x_mouth, y_mouth), r_mouth = edits.find_circle((lm[0], lm[17]), flipped_rgb.shape, 1.5)

                cv2.putText(flipped_rgb, "MOUTH OPEN", (10, flipped_rgb.shape[0] - 30), cv2.QT_FONT_NORMAL, 2,
                            (0, 255, 0),
                            thickness=2)
        else:
            self.open_mouth_confidence = max(0, self.open_mouth_confidence - 5)

        if hands:
            for hand, hand_info in zip(hands, hands_info):
                is_left = "Left" in str(hand_info)

                if is_left:
                    i = 0
                else:
                    i = 1

                if recognizer.is_correct_gesture(hand, is_left):
                    if not self.cooldown[i]:
                        flipped_rgb = edits.insert_image_in_hand(flipped_rgb, self.hotdog_image, hand, is_left)

                    (x, y), r = edits.find_circle(hand.landmark, flipped_rgb.shape, 1.25)
                    if faces and self.open_mouth_confidence > 50:
                        hand_pos = np.array([x, y])
                        mouth_pos = np.array([x_mouth, y_mouth])

                        dd_hand_mouth = np.linalg.norm(hand_pos - mouth_pos)

                        if dd_hand_mouth < r + r_mouth:
                            if not self.cooldown[i] and self.timer > 0:
                                self.cooldown[i] = True

                                self.score += 1
                        else:
                            self.cooldown[i] = False
                    else:
                        self.cooldown[i] = False

        if self.score == 0:
            text = f"Start eating hotdogs - Player #{self.player_id}"

            self.start_time = time()
        elif self.score == 1:
            text = f"{self.score} hotdog ({self.timer} seconds left) - Player #{self.player_id}"
        else:
            text = f"{self.score} hotdogs ({self.timer} seconds left) - Player #{self.player_id}"

        self.timer = self.max_time - min(self.max_time, int(time() - self.start_time))

        cv2.putText(flipped_rgb, text, (10, 50), cv2.QT_FONT_NORMAL, 1, (0, 0, 0), thickness=2)

        self.frame = flipped_rgb

    def draw(self) -> None:
        self.vm.clear_screen(COLOR_BG_MENU)

        if self.state == "eating":
            cap_size = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.vm.draw_image(((SCREEN_WIDTH - cap_size[0]) // 2, (SCREEN_HEIGHT - cap_size[1]) // 2),
                               pygame.image.frombuffer(self.frame.tostring(), self.frame.shape[1::-1],
                                                       "RGB"))
        elif self.state == "interlude":
            t1 = f"Next player #{self.player_id}"
            t2 = "Press SPACE to start"
            t3 = "Press ENTER to get results"

            tw1, th1 = self.vm.get_text_size(t1, font_size=100)
            tw2, th2 = self.vm.get_text_size(t2, font_size=50)
            tw3, th3 = self.vm.get_text_size(t3, font_size=20)

            self.vm.draw_text(
                ((SCREEN_WIDTH - tw1) // 2, (SCREEN_HEIGHT - th1 - 75) // 2),
                t1,
                COLOR_TEXT_PRIMARY,
                font_size=100,
            )

            self.vm.draw_text(
                ((SCREEN_WIDTH - tw2) // 2, (SCREEN_HEIGHT - th2 + 75) // 2),
                t2,
                COLOR_TEXT_SECONDARY,
                font_size=50
            )

            self.vm.draw_text(
                ((SCREEN_WIDTH - tw3) // 2, SCREEN_HEIGHT - th3 - 10),
                t3,
                COLOR_TEXT_SECONDARY,
                font_size=20
            )

        elif self.state == "results":
            scoreboard = list(enumerate(self.results))

            scoreboard.sort(key=lambda x: (-x[1], x[0]))

            title = "Results"
            tw, th = self.vm.get_text_size(title, font_size=110)
            self.vm.draw_text(((SCREEN_WIDTH - tw) // 2, 70), title, COLOR_TEXT_PRIMARY, font_size=110)

            start_y = 220
            line_h = 55
            font_size = 48

            for rank, (pid, score) in enumerate(scoreboard, start=1):
                line = f"{rank}) Player #{pid} - {score}"
                twl, thl = self.vm.get_text_size(line, font_size=font_size)
                self.vm.draw_text(((SCREEN_WIDTH - twl) // 2, start_y + (rank - 1) * line_h),
                                  line, COLOR_TEXT_PRIMARY, font_size=font_size)

            t = "Press ESC to return to menu"
            tw, th = self.vm.get_text_size(t, font_size=20)

            self.vm.draw_text(
                ((SCREEN_WIDTH - tw) // 2, SCREEN_HEIGHT - th - 10),
                t,
                COLOR_TEXT_SECONDARY,
                font_size=20
            )
