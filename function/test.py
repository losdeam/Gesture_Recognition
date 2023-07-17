import os
import argparse

from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img


def test(args):
    device = args.device

    net = Generator()


    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    if torch.cuda.is_available():
        try:
            device =   'cuda:0'
            net.cuda().eval()
            print("使用gpu")
            print(f"model loaded: {args.checkpoint}")
            os.makedirs(args.output_dir, exist_ok=True)

            for image_name in sorted(os.listdir(args.input_dir)):
                if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
                    continue

                image = load_image(os.path.join(args.input_dir, image_name), args.x32)

                with torch.no_grad():
                    image = to_tensor(image).unsqueeze(0) * 2 - 1
                    out = net(image.to(device), args.upsample_align).cpu()
                    out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
                    out = to_pil_image(out)

                out.save(os.path.join(args.output_dir, image_name))
                print(f"image saved: {image_name}")
        except:
            print("存在gpu但配置未达要求，已改为cpu运行")
            print(f"model loaded: {args.checkpoint}")
            device = 'cpu'
            net.to("cpu").eval()
            os.makedirs(args.output_dir, exist_ok=True)

            for image_name in sorted(os.listdir(args.input_dir)):
                if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
                    continue

                image = load_image(os.path.join(args.input_dir, image_name), args.x32)

                with torch.no_grad():
                    image = to_tensor(image).unsqueeze(0) * 2 - 1
                    out = net(image.to(device), args.upsample_align).cpu()
                    out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
                    out = to_pil_image(out)

                out.save(os.path.join(args.output_dir, image_name))
                print(f"image saved: {image_name}")
    else:
        net.to("cpu").eval()


def run(list):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=list[0],
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default=list[1],
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=list[2],
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
    )
    parser.add_argument(
        '--upsample_align',
        type=bool,
        default=False,
        help="Align corners in decoder upsampling layers"
    )
    parser.add_argument(
        '--x32',
        action="store_true",
        help="Resize images to multiple of 32"
    )
    args = parser.parse_args()
    test(args)
if __name__ == '__main__':
    list = ["weights/paprika.pt","samples\inputs","./samples/results"]
    run(list)
# python test.py --checkpoint weights/paprika.pt --input_dir samples\inputs --output_dir ./samples/results --device cpu