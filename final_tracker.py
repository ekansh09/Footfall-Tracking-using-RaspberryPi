# python final_tracker.py --weights 'F:\Game Theory\Code\yolo_counting\yolov8n.pt' --source "F:\Game Theory\Code\yolo_counting\video\sample.mp4" --save-img
import argparse
from collections import defaultdict
from pathlib import Path
import math

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)
track_dist_history = defaultdict(list)
track_status_in = defaultdict(list)
track_status_out = defaultdict(list)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

current_region = None
counting_regions = [
    {
        'name': 'YOLOv8 Polygon Region',
        'polygon': Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # Polygon points
        'counts': 0,
        'dragging': False,
        'region_color': (255, 42, 4),  # BGR Value
        'text_color': (255, 255, 255)  # Region Text Color
    }]


def mouse_callback(event, x, y, flags, param):
    """Mouse call back event."""
    global current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region['polygon'].contains(Point((x, y))):
                current_region = region
                current_region['dragging'] = True
                current_region['offset_x'] = x
                current_region['offset_y'] = y
        

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)
        if current_region is not None and current_region['dragging']:
            dx = x - current_region['offset_x']
            dy = y - current_region['offset_y']
            current_region['polygon'] = Polygon([
                (p[0] + dx, p[1] + dy) for p in current_region['polygon'].exterior.coords])
            current_region['offset_x'] = x
            current_region['offset_y'] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region['dragging']:
            current_region['dragging'] = False


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def orientation(p, q, r):
    """ 
    Find orientation of ordered triplet (p, q, r).
    Returns 0 if p, q, r are collinear, 1 if Clockwise, 2 if Counterclockwise.
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0  # collinear
    return 1 if val > 0 else 2  # clock or counterclock wise

def do_segments_intersect(l1, l2):
    """ 
    Returns True if the line segments 'p1q1' and 'p2q2' intersect.
    """
    # Find the 4 orientations required for the general and special cases
    (p1, q1), (p2, q2) = l1, l2 
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1): return True

    # p1, q1 and p2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1): return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2): return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2): return True

    # If none of the cases
    return False

def on_segment(p, q, r):
    """ 
    Given three collinear points p, q, and r, check if point q lies on line segment 'pr' 
    """
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))



def run(
    weights='yolov8n.pt',
    source=None,
    device='cpu',
    view_img=False,
    save_img=False,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be Polygons or rectangle in shape

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        classes (list): classes to detect and track
        line_thickness (int): Bounding box thickness.
        track_thickness (int): Tracking line thickness
        region_thickness (int): Region thickness.
    """
    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(f'{weights}')
    model.to('cuda') if device == '0' else model.to('cpu')

    # Extract classes names
    names = model.model.names

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    # Output setup
    save_dir = increment_path(Path('ultralytics_rc_output') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f'{Path(source).stem}.mp4'), fourcc, fps, (frame_width, frame_height))

    in_counter = 0
    out_counter = 0

    # Iterate over video frames
    while videocapture.isOpened():

        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results
        results = model.track(frame, persist=True, classes=[0])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()    
            clss = results[0].boxes.cls.cpu().tolist()

            line1 = (220,720), (220,480)
            line2 = (1000,720), (1000,480)

            cv2.line(frame, line1[0], line1[1] , (0, 255, 0), 2)
            cv2.putText(frame, 'Exit', ( line1[0][0]+ 15, line1[0][1]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), line_thickness)

            cv2.line(frame, line2[0], line2[1], (0, 255, 0), 2)
            cv2.putText(frame, 'Entry', ( line2[0][0]-80, line2[0][1]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), line_thickness)


            # annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                # annotator.box_label(box, str(names[cls]+ ' '+ str(track_id)), color=colors(cls, True))
                
                
                bbox_center = (box[0] + box[2]) / 2, (box[3])  # Bbox center

                track = track_history[track_id]  # Tracking Lines plot
                start_x = 0
                current_x = 0
                if len(track)!=0 :
                    start_x = int(track[0][0])
                    start_y = int(track[0][1])

                    current_x = int(track[-1][0])
                    current_y = int(track[-1][1])

                    p_line = (start_x, start_y), (current_x, current_y)

                    # cv2.line(frame, p_line[0], p_line[1], (0, 255, 0), 2)

                    track_dist_history[track_id] = calculate_distance(p_line[0], p_line[1])


    ### tracking should start from both side entry not from between with a buffer which can be positive or negative

                    if do_segments_intersect(line1, p_line) or do_segments_intersect(line2, p_line):
                        if track_dist_history[track_id]>50:
                            if current_x - start_x >0:
                            #     if do_segments_intersect(line1, p_line) and track_id not in track_status_in:
                            #         in_counter = in_counter + 1
                            #         track_status_in[track_id] = True
                                if do_segments_intersect(line2, p_line) and track_id not in track_status_in:
                                    in_counter = in_counter + 1
                                    track_status_in[track_id] = True
                            if current_x - start_x < 0:
                                # if do_segments_intersect(line2, p_line) and track_id not in track_status_in:
                                #     in_counter = in_counter + 1
                                #     track_status_in[track_id] = True
                                if do_segments_intersect(line1, p_line) and track_id not in track_status_out:
                                    out_counter = out_counter + 1
                                    track_status_out[track_id] = True
                
                cv2.rectangle(frame, (20, 10), (300, 50), (255,255,255), -1)
                cv2.rectangle(frame, (20, 80), (300, 120), (255,255,255), -1)
                cv2.putText(frame, 'Total In: '+ str(in_counter), (60,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), line_thickness)
                cv2.putText(frame, 'Total Out: ' + str(out_counter), (60,105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), line_thickness)

                track.append((float(bbox_center[0]), float(bbox_center[1])))

                if len(track) > 30:
                    track.pop(0)
                
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                # Check if detection inside region
                for region in counting_regions:
                    if region['polygon'].contains(Point((bbox_center[0], bbox_center[1]))):
                        region['counts'] += 1


        # # Draw regions (Polygons/Rectangles)
        # for region in counting_regions:
        #     region_label = str(region['counts'])
        #     region_color = region['region_color']
        #     region_text_color = region['text_color']

        #     polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)
        #     centroid_x, centroid_y = int(region['polygon'].centroid.x), int(region['polygon'].centroid.y)

        #     text_size, _ = cv2.getTextSize(region_label,
        #                                     cv2.FONT_HERSHEY_SIMPLEX,
        #                                     fontScale=0.7,
        #                                     thickness=line_thickness)
        #     text_x = centroid_x - text_size[0] // 2
        #     text_y = centroid_y + text_size[1] // 2
            # cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5),
            #               region_color, -1)
            # cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color,
            #             line_thickness)
        #     cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        if view_img:
            cv2.imshow('Frame',frame)
            
        if save_img:
            video_writer.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region['counts'] = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()

def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yoov8nl.pt', help='initial weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--source', type=str, required=True, help='video file path')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--line-thickness', type=int, default=2, help='bounding box thickness')
    parser.add_argument('--track-thickness', type=int, default=2, help='Tracking line thickness')
    parser.add_argument('--region-thickness', type=int, default=4, help='Region thickness')

    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)