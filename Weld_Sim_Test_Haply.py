#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
import pygame
import json
import sys
from pathlib import Path
from Physics import Physics

# ==================== Robot Model ===================== #

class robot_arm_2dof:
    def __init__(self, l):
        self.l = l 
    
    # Arm Jacobian matrix
    def Jacobian(self, q):
        J = np.array([[-self.l[0]*np.sin(q[0]) - self.l[1]*np.sin(q[0] + q[1]),
                     -self.l[1]*np.sin(q[0] + q[1])],
                    [self.l[0]*np.cos(q[0]) + self.l[1]*np.cos(q[0] + q[1]),
                     self.l[1]*np.cos(q[0] + q[1])]])
        return J
    
    # Inverse Kinematics
    def IK(self, p):
        q = np.zeros([2])
        x, y = p[0], p[1]
        r = math.hypot(x, y)
        l0 = float(self.l[0])
        l1 = float(self.l[1])
        
        def _clamp(v, lo=-1.0, hi=1.0):
            return max(lo, min(hi, v))

        denom = 2.0 * l0 * l1
        if denom == 0.0:
            q[1] = 0.0
        else:
            cos_q1 = _clamp((l0*l0 + l1*l1 - r*r) / denom)
            q[1] = math.pi - math.acos(cos_q1)

        if r < 1e-8:
            q[0] = 0.0
        else:
            numer = (l0*l0 - l1*l1 + r*r)
            denom2 = 2.0 * l0 * r
            if denom2 == 0.0:
                q0_offset = 0.0
            else:
                cos_arg = _clamp(numer / denom2)
                q0_offset = math.acos(cos_arg)
            q[0] = math.atan2(y, x) - q0_offset

        return q

# ==================== Parameters  ===================== #

dt = 0.01
dts = dt * 1

x0 = 0.0
y0 = -0.30
l1 = 0.33
l2 = 0.33
l = [l1, l2]

Ks = np.diag([200, 200])
theta = 0.0
stiffness_value_increment = 100
stiffness_angle_increment = 10 * np.pi / 180

model = robot_arm_2dof(l)

# ==================== Pygame Setup ===================== #

pygame.init()
window = pygame.display.set_mode((800, 600))
window.fill((255, 255, 255))
xc, yc = window.get_rect().center
pygame.display.set_caption('robot arm')

font = pygame.font.Font('freesansbold.ttf', 12)
text = font.render('robot arm', True, (0, 0, 0), (255, 255, 255))
textRect = text.get_rect()
textRect.topleft = (10, 10)

clock = pygame.time.Clock()
FPS = int(1 / dts)

# ==================== Initial Conditions ===================== #

t = 0.0
pm = np.zeros(2)
pr = np.zeros(2)
# Initialise arm endpoint to match HAPLY rest position (haply_screen_center - [x0,y0])
# so there is no position error on frame 1 and no jump at startup.
# haply_screen_center[1]=0.12, y0=-0.30 → p_y = 0.12-(-0.30) = 0.42 (upper screen)
p = np.array([0.0, 0.42])
dp = np.zeros(2)
F = np.zeros(2)
F_oc = np.zeros(2)
q = np.zeros(2)
p_prev = np.zeros(2)
m = 0.5
i = 0
state = []

pr_prev = p.copy()  # match arm start so dp_ref=0 on frame 1
last_dwell_print_time = 0.0
window_scale = 800
dp_ref_prev = np.zeros(2)

physics = None
device_connected = False
if Physics is not None:
    try:
        physics = Physics(hardware_version=3)
        device_connected = physics.is_device_connected()
    except Exception:
        physics = None
        device_connected = False

# ==================== Disturbance & Visual ===================== #

K = Ks.copy()
B = np.diag([20.0, 20.0])

# ==================== Path Definition (Sinusoidal) ===================== #

num_seam_segments = 50
px_vals = np.linspace(-0.4, 0.4, num_seam_segments + 1)
# Create a sine wave path that curves up and down (even lower amplitude, even higher frequency)
py_vals = 0.025 * np.sin(px_vals * 2 * np.pi / 0.1) 
seam_points = np.column_stack((px_vals, py_vals))

bot_start = p.copy()

# ==================== Disturbance ===================== #

np.random.seed(1)
dist_freqs = np.array([0.1, 0.25, 0.4, 0.7])
dist_amps = np.array([2.0, 1.0, 0.6, 0.3]) * 0.3
dist_phase = np.random.rand(len(dist_freqs)) * 2 * np.pi

def chaotic_disturbance(t):
    signal = 0.0
    for A, f, ph in zip(dist_amps, dist_freqs, dist_phase):
        signal += A * np.sin(2*np.pi*f*t + ph)
    signal_y = 0.8*signal + 0.4*np.sin(1.1*2*np.pi*t + dist_phase[0])
    return np.array([signal, signal_y])

# ==================== Ocean Current Parameters ===================== #

oc_np_waves = 4
np.random.seed(42)
oc_freqs = np.random.uniform(0.02, 0.08, oc_np_waves)
oc_amps = np.random.uniform(20.0, 50.0, oc_np_waves)
oc_wavelengths = np.random.uniform(0.6, 2.5, oc_np_waves)
oc_kvecs = np.stack([np.cos(np.random.rand(oc_np_waves)*2*np.pi), np.sin(np.random.rand(oc_np_waves)*2*np.pi)], axis=1)
oc_kvecs = oc_kvecs * (2*np.pi/oc_wavelengths)[:,None]
oc_phase = np.random.rand(oc_np_waves)*2*np.pi
oc_dirs = np.stack([np.cos(np.random.rand(oc_np_waves)*2*np.pi), np.sin(np.random.rand(oc_np_waves)*2*np.pi)], axis=1)

dir_norms = np.linalg.norm(oc_dirs, axis=1)
dir_norms[dir_norms == 0] = 1.0
oc_dirs = (oc_dirs.T / dir_norms).T

abs_mean = np.mean(np.abs(oc_dirs), axis=0)
if abs_mean[0] > 0 and abs_mean[1] > 0:
    scale_x = abs_mean[1] / abs_mean[0]
    oc_dirs[:,0] *= scale_x
    dir_norms = np.linalg.norm(oc_dirs, axis=1)
    eps = 1e-8
    for ii in range(len(dir_norms)):
        if dir_norms[ii] < eps:
            angle = np.random.rand()*2*math.pi
            oc_dirs[ii] = np.array([math.cos(angle), math.sin(angle)])
        else:
            oc_dirs[ii] = oc_dirs[ii] / dir_norms[ii]

haply_flip_x = True
haply_flip_y = True
haply_scale_x = 5.0
haply_scale_y = 5.5
haply_force_scale = 0.25
haply_force_limit = 1.2
haply_center_raw = None
haply_screen_center = np.array([0.0, 0.12], dtype=float)

oc_endpoint_scale = 0.06
oc_endpoint_scale_default = oc_endpoint_scale
oc_endpoint_enabled = True
show_true_position = True

def ocean_current_field(pos, t):
    total = np.zeros(2)
    for i in range(oc_np_waves):
        kdotx = oc_kvecs[i,0]*pos[0] + oc_kvecs[i,1]*pos[1]
        val = oc_amps[i] * math.sin(kdotx - 2*math.pi*oc_freqs[i]*t + oc_phase[i])
        total += val * oc_dirs[i]
    return total

# ==================== Visual State ===================== #

burn_threshold = 0.12
burn_intensity = 0.0

dwell_time = 0.0
dwell_speed_threshold = 0.05
dwell_seam_proximity = 0.02
dwell_decay_rate = 1.0
dwell_time_to_burn = 0.1

seam_cells = num_seam_segments
seam_burn = np.zeros(seam_cells)
seam_burn_gain = 1.0 / dwell_time_to_burn
seam_burn_decay = 0.02
weld_completion_threshold = 0.95
weld_completion_fraction = 0.98


# ==================== Main Loop ===================== #

history_dist = []
history_speed = []
history_pos = []
sim_start_time = None

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.KEYUP and event.key == ord('e'):
            run = False


i = 0
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYUP:
            if event.key == ord('q'):
                run = False
            if event.key == ord('o'):
                # toggle ocean-current-on-endpoint for debugging
                oc_endpoint_enabled = not oc_endpoint_enabled
                oc_endpoint_scale = oc_endpoint_scale_default if oc_endpoint_enabled else 0.0
                print(f"ocean-current-on-endpoint toggled: {oc_endpoint_enabled} (scale={oc_endpoint_scale})")
            if event.key == ord('t'):
                show_true_position = not show_true_position
                print(f"true-position marker toggled: {show_true_position}")
    
    # ==================== Dynamics  ===================== #

    if device_connected and physics is not None:
        try:
            _, _, _, _, pE = physics.get_device_pos()
            x_input = -pE[0] if haply_flip_x else pE[0]
            y_input = -pE[1] if haply_flip_y else pE[1]
            raw_input = np.array([x_input, y_input], dtype=float)
            if haply_center_raw is None:
                haply_center_raw = raw_input.copy()
            pr_screen = np.array([
                (raw_input[0] - haply_center_raw[0]) * haply_scale_x + haply_screen_center[0],
                (raw_input[1] - haply_center_raw[1]) * haply_scale_y + haply_screen_center[1]
            ], dtype=float)
            pm = np.array([
                pr_screen[0] * window_scale + xc,
                -pr_screen[1] * window_scale + yc
            ])
        except Exception:
            device_connected = False
            pm = np.array(pygame.mouse.get_pos())
            pr_screen = np.array([(pm[0] - xc) / window_scale, -(pm[1] - yc) / window_scale])
    else:
        pm = np.array(pygame.mouse.get_pos())
        pr_screen = np.array([(pm[0] - xc) / window_scale, -(pm[1] - yc) / window_scale])

    pr = pr_screen - np.array([x0, y0])
    
    dp_ref_raw = (pr - pr_prev)/dt
    dp_ref = 0.8 * dp_ref_prev + 0.2 * dp_ref_raw
    dp_ref_prev = dp_ref.copy()
    pr_prev = pr.copy()
    
    pos_err = pr - p
    vel_err = dp_ref - dp
    
    F_cmd = K.dot(pos_err) + B.dot(vel_err)
    F_dist = chaotic_disturbance(t)
    
    endpoint_world = p + np.array([x0, y0])
    F_oc = ocean_current_field(endpoint_world, t) * oc_endpoint_scale
    F = F_cmd + F_dist + F_oc

    if device_connected and physics is not None:
        f_haply_sim = np.clip(F_cmd * haply_force_scale, -haply_force_limit, haply_force_limit)
        f_haply_device = np.array([
            -f_haply_sim[0] if haply_flip_x else f_haply_sim[0],
            -f_haply_sim[1] if haply_flip_y else f_haply_sim[1]
        ], dtype=float)
        try:
            physics.update_force(f_haply_device)
        except Exception:
            device_connected = False

    p_prev = p.copy()

    state.append([t, pr[0], pr[1], p[0], p[1], dp[0], dp[1], F[0], F[1], K[0,0], K[1,1]])
    history_pos.append((p[0] + x0, p[1] + y0))

    if sim_start_time is None:
        sim_start_time = t

    ddp = F/m
    dp += ddp*dt
    p += dp*dt
    t += dt
    
    # ==================== Seam & Dwell  ===================== #

    endpoint_world = p + np.array([x0, y0])
    
    # Check distance to the closest segment of the polyline
    min_dist_to_seam = float('inf')
    closest = None
    best_segment_idx = 0

    for idx in range(num_seam_segments):
        a = seam_points[idx]
        b = seam_points[idx + 1]
        ap = endpoint_world - a
        ab = b - a
        ab_len2 = np.dot(ab, ab)
        t_proj = 0.0 if ab_len2 == 0.0 else np.dot(ap, ab) / ab_len2
        t_clamped = max(0.0, min(1.0, t_proj))
        candidate_closest = a + t_clamped * ab
        dist = np.linalg.norm(endpoint_world - candidate_closest)
        if dist < min_dist_to_seam:
            min_dist_to_seam = dist
            closest = candidate_closest
            best_segment_idx = idx

    dist_to_seam = min_dist_to_seam
    speed = np.linalg.norm(dp)
    
    history_dist.append(dist_to_seam)
    history_speed.append(speed)

    if speed < dwell_speed_threshold and dist_to_seam <= dwell_seam_proximity:
        dwell_time += dt
    else:
        dwell_time = max(0.0, dwell_time - dwell_decay_rate * dt)
        
    try:
        is_dwelling = (speed < dwell_speed_threshold and dist_to_seam <= dwell_seam_proximity)
        cell_idx = best_segment_idx
    except NameError:
        is_dwelling = False
        cell_idx = None

    if is_dwelling and cell_idx is not None:
        seam_burn[cell_idx] = min(1.0, seam_burn[cell_idx] + seam_burn_gain * dt)

    # End sim if seam is effectively fully welded
    welded_fraction = np.mean(seam_burn >= weld_completion_threshold)
    if welded_fraction >= weld_completion_fraction:
        print("\n WHOLE SEAM SUCCESSFULLY WELDED! ENDING SIMULATION... \n")
        run = False
        break

    i = i + 1
    
    # ==================== Kinematics & Rendering  ===================== #

    q = model.IK(p)
    x1 = l1*np.cos(q[0])
    y1 = l1*np.sin(q[0])
    x2 = x1+l2*np.cos(q[0]+q[1])
    y2 = y1+l2*np.sin(q[0]+q[1])
    
    x1w = x0 + x1
    y1w = y0 + y1
    x2w = x0 + x2
    y2w = y0 + y2
    
    # ==================== Pygame Rendering ===================== #

    window.fill((255,255,255))
    
    seam_base = np.array([30, 144, 255])
    seam_hot = np.array([255, 80, 0])
    
    for idx in range(seam_cells):
        a = seam_points[idx]
        b = seam_points[idx + 1]
        bp = seam_burn[idx]
        burn_col = (
            int(seam_base[0] * (1 - bp) + seam_hot[0] * bp),
            int(seam_base[1] * (1 - bp) + seam_hot[1] * bp),
            int(seam_base[2] * (1 - bp) + seam_hot[2] * bp)
        )
        p1 = (int(window_scale * a[0] + xc), int(-window_scale * a[1] + yc))
        p2 = (int(window_scale * b[0] + xc), int(-window_scale * b[1] + yc))
        pygame.draw.line(window, burn_col, p1, p2, 6)

    alpha = min(1.0, dwell_time / dwell_time_to_burn)
    if 'dist_to_seam' in locals() and dist_to_seam <= dwell_seam_proximity:
        seam_color = seam_base * (1 - alpha) + seam_hot * alpha
    else:
        seam_color = seam_base

    if closest is not None:
        closest_px = (int(window_scale * closest[0] + xc), int(-window_scale * closest[1] + yc))
        marker_radius = max(3, int(10 * alpha))
        marker_color = (255, 180, 0, 110)
        marker_surface = pygame.Surface(window.get_size(), pygame.SRCALPHA)
        pygame.draw.circle(marker_surface, marker_color, closest_px, marker_radius)
        pygame.draw.circle(marker_surface, (255, 120, 0, 180), closest_px, marker_radius, 1)
        window.blit(marker_surface, (0, 0))

    # ==================== Render Arm ===================== #

    ref_px = int(window_scale*(pr[0] + x0) + xc)
    ref_py = int(-window_scale*(pr[1] + y0) + yc)
    
    pygame.draw.lines(window, (0, 0, 255), False, [(int(window_scale*(x0)+xc), int(-window_scale*(y0)+yc)), (int(window_scale*(x1w)+xc), int(-window_scale*(y1w)+yc)), (int(window_scale*(x2w)+xc), int(-window_scale*(y2w)+yc))], 6)
    pygame.draw.circle(window, (0, 0, 0), (int(window_scale*(x0)+xc),int(-window_scale*(y0)+yc)), 9)
    pygame.draw.circle(window, (0, 0, 0), (int(window_scale*(x1w)+xc),int(-window_scale*(y1w)+yc)), 9)
    pygame.draw.circle(window, (255, 0, 0), (int(window_scale*(x2w)+xc),int(-window_scale*(y2w)+yc)), 5)

    if show_true_position:
        pygame.draw.circle(window, (0, 255, 0), (ref_px, ref_py), 5)

    force_scale = 50/(window_scale*(l1*l1))
    pygame.draw.line(window, (0, 255, 255), (int(window_scale*(x2w)+xc),int(-window_scale*(y2w)+yc)), (int((window_scale*(x2w)+xc)+F[0]*force_scale),int((-window_scale*(y2w)+yc-F[1]*force_scale))), 2)
    force_scale_oc = 120/(window_scale*(l1*l1))
    pygame.draw.line(window, (255, 0, 255), (int(window_scale*(x2w)+xc),int(-window_scale*(y2w)+yc)), (int((window_scale*(x2w)+xc)+F_oc[0]*force_scale_oc),int((-window_scale*(y2w)+yc-F_oc[1]*force_scale_oc))), 2)
    
    # ==================== Render Vector Field ===================== #

    def _nearest_point_on_segment(pt, a, b):
        ap = pt - a
        ab = b - a
        ab_len2 = np.dot(ab, ab)
        if ab_len2 == 0:
            return a.copy(), 0.0
        t_proj = np.dot(ap, ab)/ab_len2
        t_clamped = max(0.0, min(1.0, t_proj))
        return a + t_clamped*ab, t_clamped

    def draw_arrow(s, e, color=(180,50,50), width=2):
        pygame.draw.line(window, color, s, e, width)
        vx = e[0]-s[0]
        vy = e[1]-s[1]
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            return
        ux, uy = vx/norm, vy/norm
        px, py = -uy, ux
        left = (int(e[0]-8*ux+4*px), int(e[1]-8*uy+4*py))
        right = (int(e[0]-8*ux-4*px), int(e[1]-8*uy-4*py))
        pygame.draw.polygon(window, color, [e, left, right])

    cols = 14
    rows = 10
    vec_pixel_len = 22
    k_field = 800.0
    disturbance_scale = 0.4
    pixel_per_N = 0.6
    
    for ci in range(cols):
        for ri in range(rows):
            px = (ci + 0.5) * (window.get_width() / cols)
            py = (ri + 0.5) * (window.get_height() / rows)
            
            gx = (px - xc) / window_scale
            gy = -(py - yc) / window_scale
            gpt = np.array([gx, gy])

            F_local = ocean_current_field(gpt, t)

            magN = np.linalg.norm(F_local)
            len_px = int(min(vec_pixel_len, magN * pixel_per_N))
            if magN > 1e-6:
                dir_vec = F_local / magN
            else:
                dir_vec = np.array([0.0, 0.0])

            col_interp = min(1.0, magN/60.0)
            color = (int(200*col_interp + 50*(1-col_interp)), int(80*(1-col_interp)+80*(1-col_interp)), int(200*(1-col_interp)+50*col_interp))

            sx = int(px)
            sy = int(py)
            ex = int(sx + dir_vec[0]*len_px)
            ey = int(sy - dir_vec[1]*len_px)
            draw_arrow((sx, sy), (ex, ey), color=color, width=2)

    # ==================== Weld Burn ===================== #

    vel_error_mag = np.linalg.norm(dp_ref - dp)
    if vel_error_mag > burn_threshold:
        burn_intensity = min(1.0, burn_intensity + 0.06)
    else:
        burn_intensity = max(0.0, burn_intensity - 0.01)
        
    if burn_intensity > 0.05:
        sparks = int(6*burn_intensity)
        for s in range(sparks):
            angle = np.random.rand()*2*math.pi
            r = np.random.rand()*8
            sx = int(window_scale*x2w+xc + r*math.cos(angle))
            sy = int(-window_scale*y2w+yc + r*math.sin(angle))
            color = (255, int(180*(1-burn_intensity)), 0)
            pygame.draw.circle(window, color, (sx, sy), max(1, int(3*burn_intensity)))
    
    dwell_display = round(dwell_time, 2) if 'dwell_time' in locals() else 0.0
    dist_display = round(dist_to_seam, 3) if 'dist_to_seam' in locals() else -1.0
    text = font.render("FPS = " + str( round( clock.get_fps() ) ) + "   K = " + str( [K[0,0],K[1,1]] ) + " N/m" + "   x = " + str( np.round(p,3) ) + " m" + "   F = " + str( np.round(F,0) ) + " N" + "   F_oc=" + str(np.round(F_oc,2)) + "   F_cmd=" + str(np.round(F_cmd,2)) + "   dwell=" + str(dwell_display) + "s" + "   d2s=" + str(dist_display) , True, (0, 0, 0), (255, 255, 255))
    window.blit(text, textRect)
    
    pygame.display.flip()
    
    # ====================   Exit ===================== #

    clock.tick(FPS)
    
    if run == False:
        break

pygame.quit()

if physics is not None:
    try:
        physics.close()
    except Exception:
        pass

    # ====================   Metrics Calculation & Export ===================== #

if len(history_dist) > 0:
    # Convert recorded values from meters natively to millimeters (* 1000)
    history_dist = np.array(history_dist) * 1000.0
    history_speed = np.array(history_speed) * 1000.0
    
    # Calculate RMSE (Root Mean Square Error of the distance to the perfect weld line)
    rmse = np.sqrt(np.mean(history_dist**2))
    
    # Calculate constant velocity deviation (Standard deviation of speed)
    vel_mean = np.mean(history_speed)
    vel_std = np.std(history_speed)
    
    completion_time = round(float(t - (sim_start_time or 0.0)), 2)

    metrics = {
        "Completion_Time_s": completion_time,
        "RMSE_Position_mm": round(float(rmse), 1),
        "Velocity_Mean_mm_s": round(float(vel_mean), 1),
        "Velocity_Std_Dev_mm_s": round(float(vel_std), 1)
    }

    print("\n" + "="*30)
    print("        WELDING METRICS")
    print("="*30)
    print(f" Completion Time: {completion_time:.2f} s")
    print(f" Position RMSE : {rmse:.1f} mm")
    print(f" Mean Velocity : {vel_mean:.1f} mm/s")
    print(f" Velocity Std  : {vel_std:.1f} mm/s")
    print("="*30 + "\n")

    with open("weld_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(">> Metrics have been successfully saved to 'weld_metrics.json'")

    # ==================== Trajectory Plot ===================== #

    import os
    os.makedirs("plots", exist_ok=True)

    pos_arr = np.array(history_pos)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw seam with burn colour per segment
    seam_base_c = np.array([30, 144, 255]) / 255.0
    seam_hot_c  = np.array([255, 80,   0]) / 255.0
    for idx in range(num_seam_segments):
        a = seam_points[idx]
        b = seam_points[idx + 1]
        bp = seam_burn[idx]
        col = seam_base_c * (1 - bp) + seam_hot_c * bp
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, linewidth=4, solid_capstyle='round')

    # Draw endpoint trajectory
    ax.plot(pos_arr[:, 0], pos_arr[:, 1], color='steelblue', linewidth=1.2,
            alpha=0.7, label='Endpoint trajectory')
    ax.plot(pos_arr[0, 0],  pos_arr[0, 1],  'go', markersize=8, label='Start')
    ax.plot(pos_arr[-1, 0], pos_arr[-1, 1], 'rs', markersize=8, label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Weld Trajectory  |  RMSE {rmse:.1f} mm  |  Time {completion_time:.1f} s')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)

    plot_path = os.path.join("plots", "weld_trajectory.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f">> Trajectory plot saved to '{plot_path}'")








