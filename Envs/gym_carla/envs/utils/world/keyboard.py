import pygame

from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_ESCAPE
from pygame.locals import K_q
from pygame.locals import K_n   # next_sensor()
from pygame.locals import K_TAB # toggle_camera() : to change the location of sensors
from pygame.locals import K_c   # toggle_weather() : to change the weather
from pygame.locals import K_r   # toggle_recording() : to record the image
from pygame.locals import K_i   # toggle_info()

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        self.world = world

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_n:
                    self.world.camera_manager.next_sensor()
                elif event.key == K_TAB:
                    self.world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    self.world.next_weather(reverse=True)
                elif event.key == K_c:
                    self.world.next_weather()
                # elif event.key == K_r:
                #     self.world.camera_manager.toggle_recording()
                elif event.key == K_i:
                    self.world.hud.toggle_info()

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
