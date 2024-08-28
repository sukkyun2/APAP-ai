import os
import cv2
import time
import numpy as np
import torch
import threading
from collections import deque
from datetime import datetime
from openai import OpenAI
from func_timeout import func_set_timeout
from typing import List
from dataclasses import dataclass
from onemetric.cv.utils.iou import box_iou_batch
import supervision as sv
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics import YOLOWorld, YOLO


class CCTVMonitoringSystem:
    def __init__(self):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.TORCH_CUDA_ARCH_LIST = "8.6"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'selected device: {self.device}')

        self.client = OpenAI(
            api_key="API KEY")
        self.prompt_setting = ("{You are the monitoring manager of the stationary CCTV. \
        The input is actual data collected by CCTV cameras installed at the port, which are always fixed in a stationary position. \
        Provide key information to prevent potential or imminent risk of safety accidents. \
        Please note that the xloc and yloc represent the object location (proportional to the image), object height and width are also a proportion.}"
       "{The location information (center_x, center_y, height, width) of objects is the proportion to the image, focus especially on location information related to people or vehicles.}"
       "{Analyze the movement (speed and direction) and location (xloc and yloc) of each object to determine its trajectory. \
       Use this information to assess whether an object is moving towards a person and if so, \
       how quickly a potential safety accident might occur based on the object's speed and direction of movement.}"
       "{when you evaluate the emergency, consider the size and type of objects. \
       Given the bounding box coordinates (xyxy) of a person and the list of joint coordinates for that person \
       (nose, right eye, left eye, right ear, left ear, right shoulder, left shoulder, right elbow, left elbow, right wrist, \
       left wrist, right hip, left hip, right knee, left knee, right ankle, left ankle), analyze the spatial relationship between people and objects, \
       considering object detection information. Use this to assess the level of danger. For example, \
       if a safety helmets is not detected within person's bounding box, it can be assumed that the person is not wearing a safety helmets, \
       and the situation should be judged as 'slightly dangerous'. If the person's joint coordinates indicate a lying position, \
       it should be assessed as 'very dangerous'. If any joint coordinates are (0, 0), it means that the joint is not visually detectable, \
       and it should be excluded from consideration. Identify and report only imminent and direct threats to safety.'}")

        self.prompt_format_benchmark = 'Limit your answer into 20 words and please organize your output into this format: \
        { "danger_score": predict a score from 0 to 1 to evaluate the emergency level, non-emergency should below 0.5; \
        "reason": the main output to instant alert the CCTV Operator for emergency.}'

        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()

        self.model_yolo_world = YOLOWorld('yolov8x-worldv2.pt')
        self.model_yolo_pose = YOLO('yolov8n-pose.pt')

        self.detection_classes = [
            "shipping containers", "cranes", "forklifts", "cargo ships", "tugboats",
            "dock lines", "buoys", "life vests", "pallets", "harbor master office",
            "customs office", "container chassis", "mooring bollards", "dock gates",
            "marine fenders", "fuel tanks", "navigational lights", "cargo nets",
            "straddle carriers", "gantry cranes", "dockworkers", "fishing boats",
            "yachts", "speedboats", "oil tankers", "bulk carriers", "terminal tractors",
            "warehouse buildings", "dry docks", "loading ramps", "jetty", "berthing facilities",
            "quay walls", "storage sheds", "reefer containers", "loading arms", "anchor chains",
            "ship anchors", "dockside rails", "container handlers", "dunnage", "gangways",
            "marine radio", "weather station", "fire extinguishers", "safety helmets",
            "high-visibility vests", "security cameras", "water pumps", "first aid kits",
            "emergency exits", "port authority vehicles", "cargo straps", "shipping manifests",
            "barges", "container terminals", "port cranes", "navigation charts",
            "rope ladders", "lifeboats", "port security office", "coast guard vessels",
            "container scanners", "port trucks", "maintenance equipment", "electrical generators",
            "cargo scales", "harbor tugboats", "port signage", "mooring lines", "pilot boats",
            "water hoses", "fire hoses", "safety barriers", "rubber tires", "shipping labels",
            "marine engines", "ship propellers", "rope spools", "tide charts", "bollard pull test equipment",
            "tanker terminals", "container seals", "dockside cranes", "drillships", "power cables",
            "bulk storage tanks", "dockside warehouses", "lifebuoys", "marine compasses",
            "rope winches", "harbor cranes", "pilot ladders", "crane hooks", "marine charts",
            "dockside offices", "port lighting", "bunkering facilities", "harbor dredgers",
            "shipping containers with hazardous materials"
        ]
        self.model_yolo_world.set_classes(self.detection_classes)

        self.last_frame = None
        self.skipped_frame = 6
        self.motion_factor = 10
        self.frame_number = 0

    @func_set_timeout(10)
    def GPT_response(self, prompt):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": self.prompt_setting},
                {"role": "user", "content": prompt}
            ]
        )
        return completion

    def GPT_annotation(self, frame_info_i):
        object_info = str(frame_info_i)
        prompt = object_info + self.prompt_format_benchmark

        GPT_start_time = time.time()
        try:
            completion = self.GPT_response(prompt)
            response = completion.choices[0].message.content
            usage = completion.usage
        except:
            print('GPT time running out...')
            return None, None, None
        GPT_end_time = time.time()
        GPT_time_cost = round(GPT_end_time - GPT_start_time, 4)

        return response, GPT_time_cost, usage, prompt

    def calculate_movements(self, data_previous, tracker_id_previous, data_current, tracker_id_current):
        movements = {}
        bbox_map_previous = {tid: bbox for tid, bbox in zip(tracker_id_previous, data_previous) if tid is not None}
        bbox_map_current = {tid: bbox for tid, bbox in zip(tracker_id_current, data_current) if tid is not None}

        for tid_previous, bbox_previous in bbox_map_previous.items():
            if tid_previous in bbox_map_current:
                center_previous = ((bbox_previous[0] + bbox_previous[2]) / 2, (bbox_previous[1] + bbox_previous[3]) / 2)
                bbox_current = bbox_map_current[tid_previous]
                center_current = ((bbox_current[0] + bbox_current[2]) / 2, (bbox_current[1] + bbox_current[3]) / 2)
                dx = center_current[0] - center_previous[0]
                dy = center_current[1] - center_previous[1]
                movements[tid_previous] = (dx, dy)

        return movements

    def process_single_frame(self, frame):
        self.frame_number += 1
        results_pose = self.model_yolo_pose.predict(frame)
        results_world = self.model_yolo_world.track(frame, persist=True)
        annotated_frame = results_world[0].plot()

        boxes = results_world[0].boxes
        xywh = results_world[0].boxes.xywh
        mask = results_world[0].masks
        h, w = frame.shape[0:2]

        pose_result_bbox = results_pose[0].boxes.xyxy.cpu().numpy().tolist()
        pose_result = results_pose[0].keypoints.xy.cpu().numpy().tolist()
        temp_pose_list = []
        for i in range(len(pose_result_bbox)):
            temp_pose_list.append({"person b-box": pose_result_bbox[i], "joint coordinate": pose_result[i]})
        predictions = boxes.data.cpu().numpy()

        if len(predictions) > 0:
            if len(predictions[0]) == 7:
                boxes = predictions[:, 0:4]
                tracker_id = predictions[:, 4].astype(int)
                classes = predictions[:, 6].astype(int)
                scores = predictions[:, 5]
            else:
                boxes = predictions[:, 0:4]
                tracker_id = np.zeros(len(predictions)).astype(int)
                classes = predictions[:, 5].astype(int)
                scores = predictions[:, 4]
        else:
            return annotated_frame, None

        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=classes,
            tracker_id=tracker_id
        )

        detections = detections[detections.confidence > 0.6]

        current_frame = [tracker_id, boxes, classes, scores]

        movements = None
        if self.last_frame is not None:
            tracker_id_previous, data_previous, _, _ = self.last_frame
            tracker_id_current, data_current, _, _ = current_frame
            movements = self.calculate_movements(data_previous, tracker_id_previous, data_current, tracker_id_current)

        frame_info = []
        categorized_detections = {'frame_id': 0, 'data': []}

        for pid, box, label, score in zip(tracker_id, boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.model_yolo_world.names[int(label)]

            height = y2 - y1
            width = x2 - x1
            center_x = x1 + (width) // 2
            center_y = y1 + (height) // 2

            height = int(height / h * 100)
            width = int(width / w * 100)
            x_loc = int(center_x / w * 100)
            y_loc = int(center_y / h * 100)

            size = int(height * width / 100)

            dx, dy = 0, 0
            if movements and pid in movements:
                dx = int((movements[pid][0] / w * 100) * self.motion_factor)
                dy = int((movements[pid][1] / h * 100) * self.motion_factor)

            info = f'ID:{pid}, object_class:{class_name}, object_confidence:{score:.2f}, object_center_x:{x_loc}%, object_center_y:{y_loc}%, ' \
                   f'object_height:{height}%, object_width:{width}%, size: {size}%, object_movement: {(dx, dy)}%' \
                   f'people info : {temp_pose_list}'

            categorized_detections['data'].append(info)
            frame_info.append(info)

        self.last_frame = current_frame

        response, GPT_time_cost, usage, input_prompt = self.GPT_annotation(categorized_detections)

        if response:
            try:
                GPT_data = eval(response)
                level = GPT_data['danger_score']
                content = GPT_data['reason']
                text_1 = f"Emergency level: {level}"
                text_2 = content
                color = (0, 255 * (1 - level), 255 * level)
                cv2.putText(annotated_frame, text_1, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(annotated_frame, text_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            except:
                pass
        print(f"input_prompt : {input_prompt}")
        print(f"gpt_response : {response}")
        return annotated_frame, response
