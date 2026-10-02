"""Save one frame from the Gazebo nadir camera topic (system python with gz bindings)."""
import sys
import threading

import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gdnav_sim/frame.npy"
got = threading.Event()


def cb(msg: Image):
    if got.is_set():
        return
    arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
    np.save(out, arr)
    print("frame", msg.width, msg.height, "mean", arr.mean().round(1), "->", out)
    got.set()


node = Node()
topics = [t for t in node.topic_list() if t.endswith("nadir_cam")]
print("camera topics:", topics)
if not topics or not node.subscribe(Image, topics[0], cb):
    sys.exit("no camera topic")
if not got.wait(30):
    sys.exit("no frame in 30 s")
