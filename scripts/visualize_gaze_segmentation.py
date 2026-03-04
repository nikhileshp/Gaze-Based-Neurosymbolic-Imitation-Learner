import os
import glob
import json
import argparse
import sys
import pygame
import cv2
import numpy as np
from collections import defaultdict

# Constants
SCREEN_WIDTH = 160 * 4
SCREEN_HEIGHT = 210 * 4
CELL_BACKGROUND_DEFAULT = (40, 40, 40)

class SegmentationVisualizer:
    def __init__(self, data_dir, json_file):
        self.data_dir = data_dir
        self.segments = self._load_segments(json_file)
        self.images = sorted(glob.glob(os.path.join(data_dir, "*.png")), 
                             key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        
        # Build a map from frame_id to segment goal
        self.frame_goal_map = {}
        for seg in self.segments:
            start = seg["start_frame"]
            end = seg["end_frame"]
            goal = seg["goal"]
            for f in range(start, end + 1):
                self.frame_goal_map[f] = goal
                
        # Load gaze data (optional, for overlay)
        self.gaze_map = self._load_gaze(data_dir + ".txt")

        # Pygame Init
        pygame.init()
        self.window = pygame.display.set_mode((SCREEN_WIDTH + 300, SCREEN_HEIGHT))
        pygame.display.set_caption("Gaze Segmentation Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        
        self.current_frame_idx = 0
        self.running = True
        self.paused = False

    def _load_segments(self, json_file):
        with open(json_file, 'r') as f:
            return json.load(f)

    def _load_gaze(self, txt_file):
        gaze_map = {}
        if not os.path.exists(txt_file):
            return gaze_map
        with open(txt_file, 'r') as f:
            f.readline()
            for line in f:
                parts = line.strip().split(',')
                try:
                    fid = int(parts[0].split('_')[-1])
                    gaze_data = []
                    for x in parts[6:]:
                        if x and x.lower() != 'null':
                            try:
                                gaze_data.append(float(x.strip()))
                            except: pass
                    gaze_points = []
                    for i in range(0, len(gaze_data), 2):
                         if i+1 < len(gaze_data):
                             gaze_points.append((gaze_data[i], gaze_data[i+1]))
                    gaze_map[fid] = gaze_points
                except:
                    continue
        return gaze_map

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_RIGHT:
                        self.current_frame_idx += 1
                    elif event.key == pygame.K_LEFT:
                        self.current_frame_idx -= 1
                    elif event.key == pygame.K_q:
                        self.running = False

            if self.current_frame_idx < 0: self.current_frame_idx = 0
            if self.current_frame_idx >= len(self.images): self.current_frame_idx = len(self.images) - 1
            
            img_path = self.images[self.current_frame_idx]
            frame_id = int(os.path.basename(img_path).split('_')[-1].split('.')[0])
            
            self.render(img_path, frame_id)
            
            if not self.paused:
                self.current_frame_idx += 1
                if self.current_frame_idx >= len(self.images):
                    self.paused = True
            
            self.clock.tick(20)
            
        pygame.quit()

    def render(self, img_path, frame_id):
        self.window.fill(CELL_BACKGROUND_DEFAULT)
        
        # Image
        image = cv2.imread(img_path)
        if image is not None:
             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
             img_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
             img_surface = pygame.transform.scale(img_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
             self.window.blit(img_surface, (0, 0))
             
        # Gaze
        gaze_pts = self.gaze_map.get(frame_id, [])
        for gx, gy in gaze_pts:
            sx = int(gx * 4)
            sy = int(gy * 4)
            pygame.draw.circle(self.window, (255, 0, 0), (sx, sy), 5)
            
        # Goal Info
        goal = self.frame_goal_map.get(frame_id, "Unknown")
        
        # Locate current segment
        curr_seg = None
        for seg in self.segments:
            if seg["start_frame"] <= frame_id <= seg["end_frame"]:
                curr_seg = seg
                break
                
        # Sidebar
        start_x = SCREEN_WIDTH + 10
        y = 30
        
        txt = self.font.render(f"Frame: {frame_id}", True, "white")
        self.window.blit(txt, (start_x, y))
        y += 40
        
        color = "green"
        if "oxygen" in goal.lower(): color = "cyan"
        elif "enemy" in goal.lower(): color = "red"
        elif "surface" in goal.lower(): color = "yellow"
        
        txt = self.font.render(f"Goal: {goal}", True, color)
        self.window.blit(txt, (start_x, y))
        y += 40
        
        # Movement
        movement = curr_seg.get("movement", "none") if curr_seg else "N/A"
        txt = self.font.render(f"Move: {movement}", True, "white")
        self.window.blit(txt, (start_x, y))
        y += 40
        
        if curr_seg:
            txt = self.font.render(f"Seg: {curr_seg['start_frame']}-{curr_seg['end_frame']}", True, "gray")
            self.window.blit(txt, (start_x, y))
            y += 40
            
            # Show fixation counts
            txt = self.font.render("Fixations:", True, "white")
            self.window.blit(txt, (start_x, y))
            y += 30
            
            counts = curr_seg.get("fixation_counts", {})
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            sfont = pygame.font.SysFont("Arial", 18)
            for obj, count in sorted_counts:
                stxt = sfont.render(f"{obj}: {count}", True, "white")
                self.window.blit(stxt, (start_x + 10, y))
                y += 20

        pygame.display.flip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/seaquest/seaquest/54_RZ_2461867_Aug-11-09-35-18")
    parser.add_argument("--json_file", type=str, default="gaze_goals_verification.json")
    args = parser.parse_args()
    
    vis = SegmentationVisualizer(args.data_dir, args.json_file)
    vis.run()
