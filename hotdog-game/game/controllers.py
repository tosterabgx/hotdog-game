import os

import game.states as states
import pygame

from pathlib import Path
import mediapipe as mp


class VisualManager:
    def __init__(
            self, screen_size: tuple = (500, 500), caption: str = "Noname"
    ) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()

    def draw_line(self, start: tuple[int, int], end: tuple[int, int], color, width):
        pygame.draw.line(self.screen, color, start, end, width)

    def draw_circle(self, pos: tuple[int, int], color, radius: int) -> None:
        if len(color) == 4:
            d = radius * 2
            surf = pygame.Surface((d, d), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (radius, radius), radius)
            self.screen.blit(surf, (pos[0] - radius, pos[1] - radius))
        else:
            pygame.draw.circle(self.screen, color, pos, radius)

    def draw_rectangle(
            self, pos: tuple[int, int], width: int, height: int, color
    ) -> None:
        pygame.draw.rect(self.screen, color, pygame.Rect(*pos, width, height))

    def draw_bar(
            self,
            topleft: tuple[int, int],
            width: int,
            height: int,
            color,
            value: int,
            max_value: int = 100,
    ):
        value = max(0, min(value, max_value))
        ratio = 0 if max_value <= 0 else value / max_value
        fill_w = int(width * ratio)

        self.draw_rectangle(topleft, width, height, (60, 60, 60))
        if fill_w > 0:
            self.draw_rectangle(topleft, fill_w, height, color)

    def load_image(self, filename: str, size: tuple):
        path = Path(__file__).parent.parent / "assets" / "images" / filename
        im = pygame.image.load(str(path)).convert_alpha()
        return pygame.transform.scale(im, size)

    def draw_image(self, pos: tuple[int, int], im) -> None:
        self.screen.blit(im, pos)

    def update_screen(self) -> None:
        pygame.display.flip()

    def clear_screen(self, color: tuple = (0, 0, 0)) -> None:
        self.screen.fill(color)

    def get_text_size(self, text: str, font_size=24):
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, (0, 0, 0))

        return text_surface.get_size()

    def draw_text(self, pos: tuple[int, int], text: str, color, font_size=24):
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

        return text_surface.get_size()


class GameManager:
    def __init__(self, vm: VisualManager):
        self.vm = vm
        self.running = True

        self._states = {
            "menu": states.MainMenuState(self),
            "eating": states.EatingState(self),
            "help": states.HelpState(self),
        }

        self.state = "menu"

        self.hands_detector = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        )
        self.face_detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
        )

    @property
    def state(self):
        return self._current_state

    @state.setter
    def state(self, other: str):
        self._current_state = self._states[other]
        self._current_state.enter()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            self.state.handle_event(event)

    def update(self):
        self.state.update()

    def draw(self):
        self.state.draw()

    def close(self):
        self.face_detector.close()
        self.hands_detector.close()
