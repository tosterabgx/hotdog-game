import pygame
from game.config import FPS, SCREEN_HEIGHT, SCREEN_WIDTH
from game.controllers import GameManager, VisualManager

visuals = VisualManager((SCREEN_WIDTH, SCREEN_HEIGHT), "Hotdog Eater 3067")
game = GameManager(visuals)

while game.running:
    game.handle_events()

    game.update()
    game.draw()

    visuals.update_screen()
    visuals.clock.tick(FPS)

game.close()
pygame.quit()
