import os
import shutil
import sys
import argparse
import opengl_render_scanner

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--model-dir',
        dest='model_dir',
        help='3d models\' dir path',
        default=None,
        type=str
    )
    parser.add_argument(
        '--model-ext',
        dest='model_ext',
        help='3d model file name extension (default: obj)',
        default='obj',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for point data scaned (default: ./output)',
        default='./output',
        type=str
    )
    parser.add_argument(
        '--output-ext',
        dest='output-ext',
        help='point file name extension (default: pts),label default extension is txt',
        default='pts',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def model_to_point(args):
    for path, dir_list, file_list in os.walk(args.model_dir):
        for obj_name in file_list:
            obj_path = os.path.join(path, obj_name)
            opengl_render_scanner.circle_scan(obj_path, args.output_dir)
            shutil.move(obj_path, "/home/leon/Disk/dataset/ShapeNetCarObjScaned")
        break


if __name__ == '__main__':
    args = parse_args()
    model_to_point(args)
