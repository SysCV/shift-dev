import argparse
import os

import cv2
import numpy as np
import tqdm
import yaml

from .base import _BaseRender
from .utils import LabelViewer, UIConfig


class VideoRender(_BaseRender):
    def __init__(self, data_dir, label_path, sensor_cfg, fps, views=None):
        super().__init__(data_dir, label_path)
        self.fps = fps
        self.views = views
        self.cfg = sensor_cfg
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def render_sequence(
        self, vidoe_name, view, type, output_path="", convert_to_gray=False, alpha=0.5
    ):
        viewer = LabelViewer(UIConfig(height=800, width=1280))
        print(output_path)
        writer = cv2.VideoWriter(output_path, self.fourcc, self.fps, (1280, 800))
        frames = self.read_scalabel(vidoe_name)
        for frame in tqdm.tqdm(frames):
            frame_number = int(frame.name.split("_")[0])
            img = self.read_image(vidoe_name, frame_number, "img", view, "jpg")
            if convert_to_gray:
                img_g = np.repeat(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis], 3, 2
                )
            else:
                img_g = img
            if type == "det_insseg_2d":
                viewer.draw(
                    img_g,
                    frame,
                    with_attr=False,
                    with_box2d=False,
                    with_box3d=False,
                    with_poly2d=False,
                    with_graph=False,
                    with_rle=True,
                    alpha=alpha,
                )
            elif type == "det_2d":
                viewer.draw(
                    img_g,
                    frame,
                    with_attr=False,
                    with_box2d=True,
                    with_box3d=False,
                    with_poly2d=False,
                    with_graph=False,
                    with_rle=False,
                    alpha=alpha,
                )
            elif type == "det_3d":
                viewer.draw(
                    img_g,
                    frame,
                    with_attr=False,
                    with_box2d=False,
                    with_box3d=True,
                    with_poly2d=False,
                    with_graph=False,
                    with_rle=False,
                    alpha=alpha,
                )
            else:
                raise NotImplementedError
            rendered_img = viewer.to_numpy()
            writer.write(cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
        if self.zip is not None:
            self.zip.close()
        writer.release()


def main():
    parser = argparse.ArgumentParser(
        description="An offline label visualizer for Scalable file."
    )
    parser.add_argument("seq_id", type=str, help="Sequence to be visualized.")
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        help="Path to the image data group (.zip file or unzipped folder).",
    )
    parser.add_argument(
        "-l", "--label_path", type=str, help="Path to the label file (.json file)"
    )
    parser.add_argument(
        "-v",
        "--view",
        default="front",
        choices=["front", "left_45", "right_45", "left_90", "right_90", "left_stereo"],
        help="View of the data.",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["det_insseg_2d", "det_2d", "det_3d"],
        help="Select the label type for visualization. Detecting type from label file's name if the flag is not set.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="./output",
        type=str,
        help="Path to store the output video.",
    )
    parser.add_argument(
        "--fps",
        default=10,
        type=int,
        help="Framerate of the video.",
    )
    parser.add_argument(
        "--no_gray_background",
        action="store_false",
        help="Turn background images to white-black.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open("config/sensors.yaml"))
    render = VideoRender(
        args.data_dir, args.label_path, cfg, args.fps, views=[args.view]
    )
    os.makedirs(args.output_dir, exist_ok=True)
    if args.type is None:
        label_filename = args.label_path
        if "det_insseg_2d" in label_filename:
            vis_type = "det_insseg_2d"
        elif "det_2d" in label_filename:
            vis_type = "det_2d"
        elif "det_3d" in label_filename:
            vis_type = "det_3d"
        else:
            raise NotImplementedError
    else:
        vis_type = args.type
    video_path = os.path.join(args.output_dir, f"{args.seq_id}_{vis_type}.mp4")
    render.render_sequence(
        args.seq_id,
        args.view,
        vis_type,
        video_path,
        convert_to_gray=not args.no_gray_background,
    )


if __name__ == "__main__":
    main()
