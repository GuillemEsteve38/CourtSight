"""Video frame reading, shared by the perception wrapper modules."""

import cv2


def read_frames(path_video):
    """Read every frame of a video into memory.

    :return: (frames, fps) where frames is a list of BGR np.ndarray frames.
    """
    cap = cv2.VideoCapture(str(path_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps
