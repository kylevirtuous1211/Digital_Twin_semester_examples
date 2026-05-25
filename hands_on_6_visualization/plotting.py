import omni.ui as ui
import numpy as np
from omni.isaac.dynamic_control import _dynamic_control
import carb
import omni.kit.app
import math

# Global Configuration
prim_path = "/World/Franka/panda_rightfinger"
mass = 0.014
buffer_size = 400
V_MAX = 0.5    # m/s
KE_MAX = 0.001 # J
P_MAX = 0.003 # W/s

# Global State Variables
dc = _dynamic_control.acquire_dynamic_control_interface()
handle = _dynamic_control.INVALID_HANDLE

# data buffers
vel_buffer = [0.0] * buffer_size
ke_buffer = [0.0] * buffer_size
p_buffer = [0.0] * buffer_size
last_vel_vec = [0.0, 0.0, 0.0]

# UI components
course_win = None
vel_plot = None
ke_plot = None
p_plot = None
v_label = None
ke_label = None
p_label = None

_update_sub = None

def create_ui_window():
    """Create and initialize the UI window"""
    global course_win, vel_plot, ke_plot, p_plot, v_label, ke_label, p_label, _update_sub
    
    course_win = ui.Window("Advanced Physics Data Course", width=500, height=550)
    
    with course_win.frame:
        with ui.VStack(spacing=12, style={"background_color": 0xFF222222, "padding": 15}):
            ui.Label("Robotic Multi-Physics Analysis", height=20, style={"font_size": 16, "color": 0xFFBBBBBB})

            # Velocity Plot
            vel_plot = create_plot_section(ui.VStack(), "Linear Velocity (m/s)", V_MAX, 0xFFFFFFFF)
            
            # Kinetic Energy Plot
            ke_plot = create_plot_section(ui.VStack(), "Kinetic Energy (J)", KE_MAX, 0xFF00FFFF)

            # Power Plot
            p_plot = create_plot_section(ui.VStack(), "Instantaneous Power (W/s)", P_MAX, 0xFFFFFF00)

            # Data Overview 
            ui.Spacer(height=5)
            with ui.HStack(height=30):
                v_label = ui.Label("V: 0.00 m/s", style={"color": 0xFFFFFFFF, "font_size": 16})
                ke_label = ui.Label("KE: 0.00000 J", style={"color": 0xFF00FFFF, "font_size": 16})
                p_label = ui.Label("P: 0.000 W/s", style={"color": 0xFFFFFF00, "font_size": 16})

    _update_sub = omni.kit.app.get_app_interface().get_update_event_stream().create_subscription_to_pop(
        on_update, name="physics_course_update"
    )

def create_plot_section(parent, label, max_val, color):
    """Create a plot section and return the plot object"""
    with parent:
        with ui.VStack(height=120):
            ui.Label(label, height=15, style={"font_size": 16, "color": color})
            ui.Spacer(height=5)
            with ui.HStack():
                with ui.VStack(width=35):
                    ui.Label(str(max_val), style={"font_size": 9}); ui.Spacer(); ui.Label("0", style={"font_size": 9})
                with ui.ZStack():
                    ui.Rectangle(style={"background_color": 0xFF111111, "border_color": 0xFF444444, "border_width": 1})
                    plot = ui.Plot(ui.Type.LINE, 0.0, max_val, style={"color": color, "line_width": 1.8})
                    return plot


def on_update(e: carb.events.IEvent):
    """Update callback for physics simulation"""
    global handle, last_vel_vec, vel_buffer, ke_buffer, p_buffer
    
    dt = e.payload.get("dt", 1.0/50.0)
    if dt <= 0: return

    if handle == _dynamic_control.INVALID_HANDLE:
        handle = dc.get_rigid_body(prim_path)
        return

    vel_state = dc.get_rigid_body_linear_velocity(handle)
    if vel_state:
        curr_vel_vec = [vel_state.x, vel_state.y, vel_state.z]
        
        # call helper functions
        v_mag = calculate_velocity_magnitude(curr_vel_vec)
        ke = calculate_kinetic_energy(v_mag, mass)
        power = calculate_instantaneous_power(curr_vel_vec, last_vel_vec, dt, mass)

        #! Do not modify the code outside of the TODO sections.
        #! Update data buffers and plots with the calculated values from the helper functions.
        # TODO ###########################################################

        # update Buffers
        vel_buffer.pop(0); vel_buffer.append(v_mag)
        ke_buffer.pop(0);  ke_buffer.append(ke)
        p_buffer.pop(0);   p_buffer.append(power)

        # update plots
        vel_plot.set_data(*vel_buffer)
        ke_plot.set_data(*ke_buffer)
        p_plot.set_data(*p_buffer)

        # TODO ###########################################################
        
        # update data labels
        v_label.text = f"V: {v_mag:.3f} m/s"
        ke_label.text = f"KE: {ke:.5f} J"
        p_label.text = f"P: {power:.5f} W/s"
        
        last_vel_vec = curr_vel_vec

#! Do not modify the code outside of the TODO scetions.
#! You need to implement the three functions below to complete the course exercises.
# TODO2 ###########################################################

def calculate_velocity_magnitude(velocity_vector):
    """
    Task 1: calculate velocity magnitude
    Formula: v = sqrt(vx^2 + vy^2 + vz^2)
    """
    vx, vy, vz = velocity_vector
    v_mag = math.sqrt(vx * vx + vy * vy + vz * vz)
    return v_mag


def calculate_kinetic_energy(v_magnitude, mass):
    """
    Task 2: calculate kinetic energy
    Formula: Ek = 1/2 * m * v^2
    """
    ke = 0.5 * mass * v_magnitude * v_magnitude
    return ke


def calculate_instantaneous_power(curr_vel, last_vel, dt, mass):
    """
    Task 3: calculate instantaneous power
    Formula: P = F * v = (m * a) * v, where a = (v_curr - v_last) / dt
    """
    ax = (curr_vel[0] - last_vel[0]) / dt
    ay = (curr_vel[1] - last_vel[1]) / dt
    az = (curr_vel[2] - last_vel[2]) / dt
    power = mass * (ax * curr_vel[0] + ay * curr_vel[1] + az * curr_vel[2])
    return power

# TODO2 ###########################################################

def on_shutdown():
    """Shutdown callback"""
    global _update_sub
    _update_sub = None

# Main Execution
try:
    if 'course_win' in globals(): course_win.destroy()
except: pass
create_ui_window()
