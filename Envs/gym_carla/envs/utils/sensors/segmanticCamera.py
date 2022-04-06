import math
import weakref
import numpy as np

import carla
from carla import ColorConverter as cc

import pygame


# ==============================================================================
# -- SegmanticCamera -------------------------------------------------------------
# ==============================================================================

class SegmanticCamera(object):
    """ Class for segmantic segmantation camera """

    def __init__(self, parent_actor, hud, gamma_correction=2.2):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.frame_1 = np.empty((3,100,200))
        self.frame_0 = np.empty((3,100,200))
        self.stack = np.empty((6,100,200))
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType

        self._camera_transforms = [(carla.Transform(carla.Location(x=2.5 , z=1.5), carla.Rotation(pitch=-8.0, yaw=0.0, roll=0.0)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {'fov': '150'}],]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                # bp.set_attribute('image_size_x', str(hud.dim[0]))
                # bp.set_attribute('image_size_y', str(hud.dim[1]))
                bp.set_attribute('image_size_x', str(200))
                bp.set_attribute('image_size_y', str(200))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: SegmanticCamera._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3] # shape: H, W, C

        self.frame_1 = self.frame_0
        self.frame_0 = array[75:175,:,:].reshape((3,100,200))   # shape: C, H, W
        self.stack = np.concatenate((self.frame_0, self.frame_1), axis=0)
    
        # self.surface = pygame.surfarray.make_surface(array[:, :, ::-1].swapaxes(0, 1))
        
        if self.recording:
        # if True:
            image.save_to_disk('snapshot/%08d' % image.frame)
     