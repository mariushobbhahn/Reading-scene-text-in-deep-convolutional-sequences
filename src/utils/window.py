import cv2


class SlidingWindow:
    """
      Util to simulate a sliding window over an image.
      The step size controls movement speed of the window. The step size is interpreted as number of
      pixels by which the window will be moved in each step.
    """
    def __init__(self, step_size):
        self.step_size = step_size

    def number_of_slides(self, image):
        (img_h, img_w) = image.shape

        if img_w < img_h:
            return 0

        w = img_w - img_h
        steps = float(w) / float(self.step_size)

        return int(steps)

    def slides(self, image):
        (h, ) = image.shape

        for i in range(0, self.number_of_slides(image)):
            x = i * self.step_size
            yield image[0:, x:h]
