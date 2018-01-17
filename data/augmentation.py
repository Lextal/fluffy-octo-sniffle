import random
from PIL import ImageOps, Image


class Padding:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, imgs):
        return [ImageOps.expand(img, border=self.pad, fill=0) for img in imgs]


class RandomHorizontalFlip:
    def __call__(self, imgs):
        state = random.random() < 0.5
        return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs] if state else imgs


class Rescale:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, imgs):
        return [img.resize(self.size, Image.NEAREST) for img in imgs]


class RandomRescale:
    def __init__(self, min_scale=0.75, max_scale=1.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.range = max_scale - min_scale

    def __call__(self, imgs):
        _w, _h = imgs[0].size
        scale_factor = self.range * random.uniform(0, 1.0) + self.min_scale
        w, h = int(round(_w * scale_factor)), int(round(_h * scale_factor))
        return [img.resize((w, h), Image.NEAREST) for img in imgs]


class RandomCrop:
    def __init__(self, crop=(256, 256)):
        self.crop = crop

    def __call__(self, imgs):
        w, h = self.crop
        _w, _h = imgs[0].size
        x1, y1 = random.randint(0, _w - self.crop[0]), random.randint(0, _h - self.crop[1])
        return [img.crop((x1, y1, x1 + w, y1 + h)) for img in imgs]


class RandomRotation:
    def __init__(self, degree=10):
        self.degree = degree

    def __call__(self, imgs):
        deg = random.randint(-self.degree, self.degree)
        return [img.rotate(deg) for img in imgs]
