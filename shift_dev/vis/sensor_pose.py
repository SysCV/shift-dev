import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base import _BaseRender


class SensorPoseRender(_BaseRender):
    def __init__(self, label_path):
        super().__init__(None, label_path)
        self.fig = None
        self.ax = None
        self.camera_points = self._get_camera_points()
        self.color_mapper = None
        self.z_ratio = 0.2

    def _get_camera_points(self):
        # Get the camera points at its initial pose.
        # The initial pose of camera locates at the origin and points toward x-pos.
        camera_points = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
                [1, -1, 1],
                [0, 0, 0],
                [-1, -1, 1],
                [0, 0, 0],
                [-1, 1, 1],
            ],
            dtype=np.float64,
        ).transpose()
        rot_y_90 = R.from_euler("y", 90, degrees=True).as_matrix()
        camera_points = rot_y_90 @ camera_points
        return camera_points

    def _get_ax(self):
        if self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.ax.set_box_aspect((1, 1, self.z_ratio))
            self.ax.invert_xaxis()
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")
        return self.ax

    def _get_color(self, x, cmap_name):
        if self.color_mapper is None:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            self.color_mapper = matplotlib.cm.ScalarMappable(norm, cmap_name)
        color = self.color_mapper.to_rgba(x)
        return color

    def plot_camera(self, translation, rotation, scale=2.0, add_mark=False, **kwarg):
        rot_camera = rotation.transpose()
        tran_camera = (-rot_camera @ translation).reshape(3, 1)

        # translate the camera points to the world coor system
        camera_points_in_world = (
            rot_camera @ (scale * self.camera_points) + np.tile(tran_camera, (1, 12))
        ).transpose()
        self._get_ax().plot(
            xs=camera_points_in_world[:, 0],
            ys=camera_points_in_world[:, 1],
            zs=camera_points_in_world[:, 2],
            **kwarg,
        )
        if add_mark:
            self._get_ax().scatter(
                xs=camera_points_in_world[0, 0],
                ys=camera_points_in_world[0, 1],
                zs=camera_points_in_world[0, 2],
                s=30,
                c="k",
            )

    def render_sequence(
        self, vidoe_name, use_degrees=False, every_n_frame=10, cmap="rainbow_r", **kwarg
    ):
        frames = self.read_scalabel(vidoe_name)
        for i, frame in enumerate(frames):
            if i % every_n_frame == 0:
                loc = np.array(frame.extrinsics.location)
                rot = np.array(frame.extrinsics.rotation)
                rot_matrix = R.from_euler("xyz", rot, use_degrees).as_matrix()
                self.plot_camera(
                    loc,
                    rot_matrix,
                    color=self._get_color(i / len(frames), cmap),
                    add_mark=(i == 0),
                    **kwarg,
                )

        # make the aspect ratio to be equal for all axis
        scale_xy = np.array(
            [getattr(self._get_ax(), "get_{}lim".format(dim))() for dim in "xy"]
        )
        range_xy = [np.min(scale_xy), np.max(scale_xy)]
        self._get_ax().auto_scale_xyz(
            range_xy, range_xy, (0, np.max(scale_xy) * self.z_ratio)
        )
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="An offline label visualizer for sensor poses."
    )
    parser.add_argument("seq_id", type=str, help="Sequence to be visualized.")
    parser.add_argument(
        "-l", "--label_path", type=str, help="Path to the label file (.json file)"
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="If set, use degrees as the angle unit in the extrinsic, otherwise radius",
    )
    args = parser.parse_args()

    render = SensorPoseRender(args.label_path)
    render.render_sequence(args.seq_id, use_degrees=args.degrees)


if __name__ == "__main__":
    main()
