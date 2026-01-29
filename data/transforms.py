from PIL import Image


class ResizePad:
    def __init__(self, target_height, target_width, pad_value=255):
        self.target_height = target_height
        self.target_width = target_width
        self.pad_value = pad_value

    def __call__(self, img):
        w, h = img.size
        scale = min(
            self.target_width / w,
            self.target_height / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        img = img.resize((new_w, new_h), Image.BILINEAR)

        padded = Image.new(
            "L",
            (self.target_width, self.target_height),
            color=self.pad_value
        )

        left = (self.target_width - new_w) // 2
        top = (self.target_height - new_h) // 2

        padded.paste(img, (left, top))
        return padded

