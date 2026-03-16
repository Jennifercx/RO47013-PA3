#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Control in Human-Robot Interaction Assignment 2a: teleoperation & tele-impedance
-------------------------------------------------------------------------------
DESCRIPTION:
2-DOF planar robot arm model with shoulder and elbow joints. The code includes
simulation environment and visualisation of the robot.

Assume that the robot is a torque-controlled robot and its dynamics is
compensated internally, therefore the program operates at the endpoint in
Cartesian space. The input to the robot is the desired endpoint force and
the output from the robot is the measured endpoint position.

Important variables:
pm[0] -> mouse x position
pm[1] -> mouse y position
pr[0] -> reference endpoint x position
pr[1] -> reference endpoint y position
p[0] -> actual endpoint x position
p[1] -> actual endpoint y position
dp[0] -> endpoint x velocity
dp[1] -> endpoint y velocity
F[0] -> endpoint x force
F[1] -> endpoint y force

NOTE: Keep the mouse position inside the robot workspace before pressing 'e'
to start and then maintain it within the workspace during the operation,
in order for the inverse kinematics calculation to work properly.
-------------------------------------------------------------------------------


INSTURCTOR: Luka Peternel
e-mail: l.peternel@tudelft.nl

"""



import numpy as np
import math
import matplotlib.pyplot as plt
import pygame





'''ROBOT MODEL'''

class robot_arm_2dof:
    def __init__(self, l):
        self.l = l # link length
    
    
    
    # arm Jacobian matrix
    def Jacobian(self, q):
        J = np.array([[-self.l[0]*np.sin(q[0]) - self.l[1]*np.sin(q[0] + q[1]),
                     -self.l[1]*np.sin(q[0] + q[1])],
                    [self.l[0]*np.cos(q[0]) + self.l[1]*np.cos(q[0] + q[1]),
                     self.l[1]*np.cos(q[0] + q[1])]])
        return J
    
    
    
    # inverse kinematics
    def IK(self, p):
        q = np.zeros([2])
        # handle r near zero and numerical issues robustly
        x, y = p[0], p[1]
        r = math.hypot(x, y)
        l0 = float(self.l[0])
        l1 = float(self.l[1])
        # clamp helper
        def _clamp(v, lo=-1.0, hi=1.0):
            return max(lo, min(hi, v))

        # compute q1 using law of cosines (clamp argument)
        denom = 2.0 * l0 * l1
        if denom == 0.0:
            q[1] = 0.0
        else:
            cos_q1 = _clamp((l0*l0 + l1*l1 - r*r) / denom)
            # safe acos
            q[1] = math.pi - math.acos(cos_q1)

        # compute q0; guard against r==0
        if r < 1e-8:
            # if at origin, set a default configuration (arm stretched along x)
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






'''SIMULATION'''

# SIMULATION PARAMETERS
dt = 0.01 # intergration step timedt = 0.01 # integration step time
dts = dt*1 # desired simulation step time (NOTE: it may not be achieved)



# ROBOT PARAMETERS
x0 = 0.0 # base x position
# move base further down visually
y0 = -0.30 # base y position (visual offset downwards)
l1 = 0.33 # link 1 length
l2 = 0.33 # link 2 length (includes hand)
l = [l1, l2] # link length



# IMPEDANCE CONTROLLER PARAMETERS
Ks = np.diag([200,200]) # stiffness in the endpoint stiffness frame [N/m] (balanced X/Y)
theta = 0.0 # roation of the endpoint stiffness frame wrt the robot base frame [rad]
stiffness_value_increment = 100 # for tele-impedance [N/m]
stiffness_angle_increment = 10*np.pi/180 # for tele-impedance [rad]



# SIMULATOR
# initialise robot model class
model = robot_arm_2dof(l)

# initialise real-time plot with pygame
pygame.init() # start pygame
window = pygame.display.set_mode((800, 600)) # create a window (size in pixels)
window.fill((255,255,255)) # white background
xc, yc = window.get_rect().center # window center
pygame.display.set_caption('robot arm')

font = pygame.font.Font('freesansbold.ttf', 12) # printing text font and font size
text = font.render('robot arm', True, (0, 0, 0), (255, 255, 255)) # printing text object
textRect = text.get_rect()
textRect.topleft = (10, 10) # printing text position with respect to the top-left corner of the window

clock = pygame.time.Clock() # initialise clock
FPS = int(1/dts) # refresh rate

# initial conditions
t = 0.0 # time
pm = np.zeros(2) # mouse position
pr = np.zeros(2) # reference endpoint position
# starting position of the welding bot (m) - keep endpoint near center
p = np.array([0.1,0.1]) # actual endpoint position
dp = np.zeros(2) # actual endpoint velocity
F = np.zeros(2) # endpoint force
F_oc = np.zeros(2)
q = np.zeros(2) # joint position
p_prev = np.zeros(2) # previous endpoint position
m = 0.5 # endpoint mass
i = 0 # loop counter
state = [] # state vector

# previous reference for velocity calculation when using mouse control
pr_prev = pr.copy()

# scaling
window_scale = 800 # conversion from meters to pixles


# ============== Disturbance and visual state parameters ============== #

# Controller state and parameters
K = Ks.copy()  # current stiffness matrix in endpoint frame
B = np.diag([20.0, 20.0])  # damping in endpoint frame

# Path logic (training only)
# place the welding seam in the middle and make it longer
training_start = np.array([-0.4, 0.0])
training_end = np.array([0.4, 0.0])

# bot starting position (used for reset)
bot_start = p.copy()

# Disturbance (chaotic current) parameters: sum of sines for unpredictable disturbance
np.random.seed(1)
dist_freqs = np.array([0.5, 1.3, 2.1, 3.7])  # Hz
dist_amps = np.array([2.0, 1.0, 0.6, 0.3]) * 0.5  # N (per axis scaling)
dist_phase = np.random.rand(len(dist_freqs))*2*np.pi

def chaotic_disturbance(t):
    """Return a 2D disturbance force as a sum of sine waves with different
    frequencies and amplitudes. This creates an unpredictable (but repeatable)
    disturbance signal in the workspace."""
    signal = 0.0
    for A, f, ph in zip(dist_amps, dist_freqs, dist_phase):
        signal += A * np.sin(2*np.pi*f*t + ph)
    # create 2D vector by using a phase-shift for second axis
    signal_y = 0.8*signal + 0.4*np.sin(1.1*2*np.pi*t + dist_phase[0])
    return np.array([signal, signal_y])

# Ocean current / spatial force field: sum of slow, large traveling sine waves over x,y,t
# Fewer waves, larger amplitudes, longer wavelengths and lower temporal frequencies for a "chill" field
oc_np_waves = 4
np.random.seed(42)
# much slower temporal variation (Hz)
oc_freqs = np.random.uniform(0.06, 0.24, oc_np_waves)
# bigger amplitudes (N)
oc_amps = np.random.uniform(30.0, 80.0, oc_np_waves)
# spatial wavevectors (kx,ky) in rad/m - choose random directions and longer wavelengths
oc_wavelengths = np.random.uniform(0.6, 2.5, oc_np_waves)  # meters
oc_kvecs = np.stack([np.cos(np.random.rand(oc_np_waves)*2*np.pi), np.sin(np.random.rand(oc_np_waves)*2*np.pi)], axis=1)
oc_kvecs = oc_kvecs * (2*np.pi/oc_wavelengths)[:,None]
oc_phase = np.random.rand(oc_np_waves)*2*np.pi
oc_dirs = np.stack([np.cos(np.random.rand(oc_np_waves)*2*np.pi), np.sin(np.random.rand(oc_np_waves)*2*np.pi)], axis=1)
# normalize direction vectors to unit length to avoid accidental axis bias
dir_norms = np.linalg.norm(oc_dirs, axis=1)
# avoid division by zero
dir_norms[dir_norms == 0] = 1.0
oc_dirs = (oc_dirs.T / dir_norms).T
# compute mean absolute components and scale axes to balance X/Y influence
abs_mean = np.mean(np.abs(oc_dirs), axis=0)
if abs_mean[0] > 0 and abs_mean[1] > 0:
    # scale X components so that mean absolute X equals mean absolute Y
    scale_x = abs_mean[1] / abs_mean[0]
    oc_dirs[:,0] *= scale_x
    # renormalize rows and guard degenerate rows
    dir_norms = np.linalg.norm(oc_dirs, axis=1)
    eps = 1e-8
    for ii in range(len(dir_norms)):
        if dir_norms[ii] < eps:
            angle = np.random.rand()*2*math.pi
            oc_dirs[ii] = np.array([math.cos(angle), math.sin(angle)])
        else:
            oc_dirs[ii] = oc_dirs[ii] / dir_norms[ii]
    print(f"oc_dirs balanced (scale_x={scale_x:.3f}), abs_mean before scale={abs_mean}")
# scale to apply ocean current to the endpoint dynamics (set to 0 to disable)
oc_endpoint_scale = 0.1
# keep a default and an enabled flag so we can toggle at runtime for debugging
oc_endpoint_scale_default = oc_endpoint_scale
oc_endpoint_enabled = True
# print oc_dirs for a quick sanity check (will appear in console)
print("oc_dirs:", oc_dirs)

def ocean_current_field(pos, t):
    """Compute a 2D ocean-like current vector at position pos (meters) and time t."""
    total = np.zeros(2)
    for i in range(oc_np_waves):
        kdotx = oc_kvecs[i,0]*pos[0] + oc_kvecs[i,1]*pos[1]
        val = oc_amps[i] * math.sin(kdotx - 2*math.pi*oc_freqs[i]*t + oc_phase[i])
        total += val * oc_dirs[i]
    return total

# visual state for burn-through effect
burn_threshold = 0.12  # m/s velocity error threshold
burn_intensity = 0.0



# wait until the start button is pressed
run = True
while run:
    for event in pygame.event.get(): # interrupt function
        if event.type == pygame.KEYUP:
            if event.key == ord('e'): # enter the main loop after 'e' is pressed
                run = False



# MAIN LOOP
i = 0
run = True
while run:
    for event in pygame.event.get(): # interrupt function
        if event.type == pygame.QUIT: # force quit with closing the window
            run = False
        elif event.type == pygame.KEYUP:
            if event.key == ord('q'): # force quit with q button
                run = False
            if event.key == ord('o'):
                # toggle ocean-current-on-endpoint for debugging
                oc_endpoint_enabled = not oc_endpoint_enabled
                oc_endpoint_scale = oc_endpoint_scale_default if oc_endpoint_enabled else 0.0
                print(f"ocean-current-on-endpoint toggled: {oc_endpoint_enabled} (scale={oc_endpoint_scale})")
            if event.key == ord('p'):
                # print detailed ocean-field diagnostics for current endpoint
                endpoint_world = p + np.array([x0, y0])
                print("--- ocean field diagnostics ---")
                print("endpoint_world:", endpoint_world)
                print("oc_kvecs:")
                print(oc_kvecs)
                print("oc_dirs:")
                print(oc_dirs)
                print("oc_amps:")
                print(oc_amps)
                print("oc_freqs:")
                print(oc_freqs)
                print("oc_phase:")
                print(oc_phase)
                # per-wave contribution
                for ii in range(oc_np_waves):
                    kdotx = oc_kvecs[ii,0]*endpoint_world[0] + oc_kvecs[ii,1]*endpoint_world[1]
                    val = oc_amps[ii] * math.sin(kdotx - 2*math.pi*oc_freqs[ii]*t + oc_phase[ii])
                    contrib = val * oc_dirs[ii]
                    print(f"wave {ii}: kdotx={kdotx:.4f}, val={val:.4f}, dir={oc_dirs[ii]}, contrib={contrib}")
                print("F_oc total:", F_oc)
            # tele-impedance interface / control mode switch
            if event.key == ord('t'):
                # toggle training / transfer phase manually
                phase = 'transfer' if phase == 'training' else 'training'
            if event.key == ord('=') or event.key == ord('+'):
                # increase stiffness
                K += np.diag([stiffness_value_increment, stiffness_value_increment/5.0])
            if event.key == ord('-'):
                # decrease stiffness (keep positive)
                K = np.maximum(K - np.diag([stiffness_value_increment, stiffness_value_increment/5.0]), 1.0)
            if event.key == ord('r'):
                # reset robot position and dynamics
                p[:] = bot_start.copy()
                dp[:] = 0.0
                burn_intensity = 0.0
    
    
    
    '''*********** Student should fill in ***********'''
    # main control code
    # update mouse position (pixel coords)
    pm = np.array(pygame.mouse.get_pos())
    # convert mouse pixel to world (meters) and use as reference endpoint
    # map mouse to the robot base (kinematic) frame by subtracting the visual base offset
    pr_screen = np.array([ (pm[0] - xc)/window_scale, -(pm[1] - yc)/window_scale ])
    pr = pr_screen - np.array([x0, y0])
    # reference velocity is derivative of mouse position (numerical)
    dp_ref = (pr - pr_prev)/dt
    pr_prev = pr.copy()
    # simple impedance controller in endpoint frame
    pos_err = pr - p
    vel_err = dp_ref - dp
    # compute commanded force and add chaotic disturbance
    F_cmd = K.dot(pos_err) + B.dot(vel_err)
    F_dist = chaotic_disturbance(t)
    # compute ocean current at the endpoint world position (p is in base frame, add visual/kinematic base)
    endpoint_world = p + np.array([x0, y0])
    F_oc = ocean_current_field(endpoint_world, t) * oc_endpoint_scale
    # total force: controller + time-only disturbance + spatial ocean current at endpoint
    F = F_cmd + F_dist + F_oc



	# previous endpoint position for velocity calculation
    p_prev = p.copy()

    # log states for analysis
    state.append([t, pr[0], pr[1], p[0], p[1], dp[0], dp[1], F[0], F[1], K[0,0], K[1,1]])
    
    # integration
    ddp = F/m
    dp += ddp*dt
    p += dp*dt
    t += dt
    
    '''*********** Student should fill in ***********'''
    # simulate a wall
    # simple wall along x between y = -0.02 and y = 0.02 at x = 0.35 (pipe edge)
    wall_x = 0.35
    wall_y_min = -0.03
    wall_y_max = 0.03
    if p[0] > wall_x and wall_y_min <= p[1] <= wall_y_max:
        # reflect and reduce velocity to simulate contact
        dp[0] = -0.3*dp[0]
        p[0] = wall_x
        # small reaction force for logging
        F = np.array([-30.0, 0.0])

    # increase loop counter
    i = i + 1
    
    
    
    # update individual link position
    q = model.IK(p)
    x1 = l1*np.cos(q[0])
    y1 = l1*np.sin(q[0])
    x2 = x1+l2*np.cos(q[0]+q[1])
    y2 = y1+l2*np.sin(q[0]+q[1])
    # apply visual base offset to joint positions for drawing
    # kinematics (x1,y1,x2,y2) are computed in the robot base frame (origin at x0,y0)
    # add x0,y0 so the drawn positions match the visual base location without changing kinematics
    x1w = x0 + x1
    y1w = y0 + y1
    x2w = x0 + x2
    y2w = y0 + y2
    
    # real-time plotting
    window.fill((255,255,255)) # clear window
    # draw workspace: pipe / weld seam as a thick curved segment
    # draw straight pipe body
    seam_color = (30, 144, 255)
    seam_thickness = 8
    # draw training seam (straight)
    start_pix = (int(window_scale*training_start[0]+xc), int(-window_scale*training_start[1]+yc))
    end_pix = (int(window_scale*training_end[0]+xc), int(-window_scale*training_end[1]+yc))
    pygame.draw.line(window, seam_color, start_pix, end_pix, seam_thickness)
    # (transfer seam removed - training only)
    # draw reference position using world-to-screen mapping so it aligns with other world elements
    # draw reference position in screen pixels: convert from base frame back to screen by adding visual base offset
    ref_px = int(window_scale*(pr[0] + x0) + xc)
    ref_py = int(-window_scale*(pr[1] + y0) + yc)
    pygame.draw.circle(window, (0, 255, 0), (ref_px, ref_py), 5) # draw reference position
    # draw links (use world positions that include the visual base offset)
    pygame.draw.lines(window, (0, 0, 255), False, [(int(window_scale*(x0)+xc), int(-window_scale*(y0)+yc)), (int(window_scale*(x1w)+xc), int(-window_scale*(y1w)+yc)), (int(window_scale*(x2w)+xc), int(-window_scale*(y2w)+yc))], 6) # draw links
    pygame.draw.circle(window, (0, 0, 0), (int(window_scale*(x0)+xc),int(-window_scale*(y0)+yc)), 9) # draw shoulder / base
    pygame.draw.circle(window, (0, 0, 0), (int(window_scale*(x1w)+xc),int(-window_scale*(y1w)+yc)), 9) # draw elbow
    pygame.draw.circle(window, (255, 0, 0), (int(window_scale*(x2w)+xc),int(-window_scale*(y2w)+yc)), 5) # draw hand / endpoint

    force_scale = 50/(window_scale*(l1*l1)) # scale for displaying force vector
    pygame.draw.line(window, (0, 255, 255), (int(window_scale*(x2w)+xc),int(-window_scale*(y2w)+yc)), (int((window_scale*(x2w)+xc)+F[0]*force_scale),int((-window_scale*(y2w)+yc-F[1]*force_scale))), 2) # draw endpoint force vector
    # draw ocean-current force at endpoint as a magenta arrow (separate visual)
    force_scale_oc = 120/(window_scale*(l1*l1))
    pygame.draw.line(window, (255, 0, 255), (int(window_scale*(x2w)+xc),int(-window_scale*(y2w)+yc)), (int((window_scale*(x2w)+xc)+F_oc[0]*force_scale_oc),int((-window_scale*(y2w)+yc-F_oc[1]*force_scale_oc))), 2)
    
    # draw a vector field (grid of small arrows) showing the local seam tangent
    def _nearest_point_on_segment(pt, a, b):
        # project pt onto segment ab
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

    # grid in screen pixels (avoid too dense)
    cols = 14
    rows = 10
    vec_pixel_len = 22
    # virtual field parameters for grid arrows
    k_field = 800.0  # N/m pulling toward the seam
    disturbance_scale = 0.6  # scale chaotic disturbance contribution to arrows
    pixel_per_N = 0.6  # how many pixels per Newton for display scaling
    for ci in range(cols):
        for ri in range(rows):
            # compute world coordinate of grid point
            px = (ci + 0.5) * (window.get_width() / cols)
            py = (ri + 0.5) * (window.get_height() / rows)
            # convert to meters, account for center and y-flip
            gx = (px - xc) / window_scale
            gy = -(py - yc) / window_scale
            gpt = np.array([gx, gy])

            # nearest point on the training segment
            pt_on, _ = _nearest_point_on_segment(gpt, training_start, training_end)

            # use the ocean current field (spatial+temporal sine waves)
            F_local = ocean_current_field(gpt, t)

            # convert force to pixel length (clamped)
            magN = np.linalg.norm(F_local)
            len_px = int(min(vec_pixel_len, magN * pixel_per_N))
            if magN > 1e-6:
                dir_vec = F_local / magN
            else:
                dir_vec = np.array([0.0, 0.0])

            # choose color based on magnitude (blue small -> red large)
            col_interp = min(1.0, magN/60.0)
            color = (int(200*col_interp + 50*(1-col_interp)), int(80*(1-col_interp)+80*(1-col_interp)), int(200*(1-col_interp)+50*col_interp))

            sx = int(px)
            sy = int(py)
            ex = int(sx + dir_vec[0]*len_px)
            ey = int(sy - dir_vec[1]*len_px)
            draw_arrow((sx, sy), (ex, ey), color=color, width=2)

    # velocity error and burn-through visual feedback
    vel_error_mag = np.linalg.norm(dp_ref - dp)
    if vel_error_mag > burn_threshold:
        burn_intensity = min(1.0, burn_intensity + 0.02)
    else:
        burn_intensity = max(0.0, burn_intensity - 0.01)
    # if burning, draw sparks / broken weld markers near the seam point closest to endpoint
    if burn_intensity > 0.05:
        sparks = int(6*burn_intensity)
        for s in range(sparks):
            angle = np.random.rand()*2*math.pi
            r = np.random.rand()*8
            sx = int(window_scale*x2w+xc + r*math.cos(angle))
            sy = int(-window_scale*y2w+yc + r*math.sin(angle))
            color = (255, int(180*(1-burn_intensity)), 0)
            pygame.draw.circle(window, color, (sx, sy), max(1, int(3*burn_intensity)))
    
    # print data (include ocean-current and commanded force components for debugging)
    text = font.render("FPS = " + str( round( clock.get_fps() ) ) + "   K = " + str( [K[0,0],K[1,1]] ) + " N/m" + "   x = " + str( np.round(p,3) ) + " m" + "   F = " + str( np.round(F,0) ) + " N" + "   F_oc=" + str(np.round(F_oc,2)) + "   F_cmd=" + str(np.round(F_cmd,2)), True, (0, 0, 0), (255, 255, 255))
    window.blit(text, textRect)
    
    pygame.display.flip() # update display
    
    
    
    # try to keep it real time with the desired step time
    clock.tick(FPS)
    
    if run == False:
        break

pygame.quit() # stop pygame












'''ANALYSIS'''

state = np.array(state)


plt.figure(3)
plt.subplot(411)
plt.title("VARIABLES")
plt.plot(state[:,0],state[:,1],"b",label="x")
plt.plot(state[:,0],state[:,2],"r",label="y")
plt.legend()
plt.ylabel("pr [m]")

plt.subplot(412)
plt.plot(state[:,0],state[:,3],"b")
plt.plot(state[:,0],state[:,4],"r")
plt.ylabel("p [m]")

plt.subplot(413)
plt.plot(state[:,0],state[:,7],"b")
plt.plot(state[:,0],state[:,8],"r")
plt.ylabel("F [N]")

plt.subplot(414)
plt.plot(state[:,0],state[:,9],"c")
plt.plot(state[:,0],state[:,10],"m")
plt.ylabel("K [N/m]")
plt.xlabel("t [s]")

plt.tight_layout()




plt.figure(4)
plt.title("ENDPOINT BEHAVIOUR")
plt.plot(0,0,"ok",label="shoulder")
plt.plot(state[:,1],state[:,2],"lime",label="reference")
plt.plot(state[:,3],state[:,4],"r",label="actual")
plt.axis('equal')
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()

plt.tight_layout()





