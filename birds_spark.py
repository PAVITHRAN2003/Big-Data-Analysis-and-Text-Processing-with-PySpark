import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from PIL import Image
from pyspark.sql import SparkSession

def compute_speed(velocity):
    return np.linalg.norm(velocity)

def limit_speed(velocity, min_speed, max_speed):
    speed = compute_speed(velocity)
    if speed < 1e-10:
        return np.zeros_like(velocity)
    if speed < min_speed:
        velocity = velocity / speed * min_speed
    elif speed > max_speed:
        velocity = velocity / speed * max_speed
    return velocity

# Update the lead bird's position following a parametric trajectory
def update_lead_bird_position(t):
    angle = lead_bird_speed * t / lead_bird_radius
    x = lead_bird_radius * np.cos(angle)
    y = lead_bird_radius * np.sin(angle) * np.cos(angle)
    z = lead_bird_radius * (1 + 0.5 * np.sin(angle / 5))
    return np.array([x, y, z])

def compute_forces(bird_position, positions):
    distances = np.linalg.norm(positions - bird_position, axis=1)
    d_lead = distances[0]
    lead_force = (positions[0] - bird_position) * (1 / d_lead) if d_lead > 10 else np.zeros(3)

    nearest_idx = np.argmin(distances)
    d_near = distances[nearest_idx]
    cohesion_force = np.nan_to_num((positions[nearest_idx] - bird_position) * ((d_near / 1) ** 2)) if d_near > max_distance else np.zeros(3)

    close_neighbors = positions[distances < min_distance]
    close_distances = distances[distances < min_distance]
    separation_force = np.sum([(bird_position - neighbor) / (dist ** 2)
                                for neighbor, dist in zip(close_neighbors, close_distances) if dist > 0],
                               axis=0) if len(close_neighbors) > 0 else np.zeros(3)

    total_weight = np.sum([1 / ((dist / 1) ** 2) for dist in close_distances if dist > 0])
    if total_weight > 0:
        separation_force = separation_force / total_weight

    return cohesion_force + separation_force + lead_force

# Save a single frame as an image
def save_frame(positions, frame_number):
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    
    ax.scatter(positions[1:, 0], positions[1:, 1], positions[1:, 2], label="Flock Birds")
    
    # Highlight the lead bird as a star
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='red', marker='*', s=200, label="Lead Bird (Star)")
    
  
    ax.set_xlim([-lead_bird_radius, lead_bird_radius])
    ax.set_ylim([-lead_bird_radius, lead_bird_radius])
    ax.set_zlim([0, 2 * lead_bird_radius])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc="upper right")
    

    os.makedirs('./plot', exist_ok=True)
    plt.savefig(f'./plot/frame_{frame_number:03d}.png')
    plt.close()


# Create a GIF from saved frames
def create_gif():
    frames = []
    plot_dir = './plot'
    for filename in sorted(os.listdir(plot_dir)):
        if filename.endswith('.png'):
            filepath = os.path.join(plot_dir, filename)
            frame = Image.open(filepath)
            frames.append(frame)
    
    if frames:
        frames[0].save(
            'bird_simulation.gif',
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )

# Main function
if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder.appName("BirdSimulation").getOrCreate()
    
    # Simulation parameters
    num_birds = 10000
    num_frames = 500
    time_step = 1 / 4
    std_dev_position = 10.0
    lead_bird_speed = 20.0
    lead_bird_radius = 300.0
    min_speed = 10.0
    max_speed = 30.0
    max_distance = 20.0
    min_distance = 10.0

    # Initialize positions and velocities
    positions = np.random.normal(loc=np.array([0, 0, 1.5 * lead_bird_radius]), 
                                  scale=std_dev_position, 
                                  size=(num_birds, 3))
    velocities = np.zeros((num_birds, 3))

    simulation = []
    time_cost = []
    
    for frame in range(num_frames):
        start = time.time()
        
        # Update lead bird position
        positions[0] = update_lead_bird_position(frame * time_step)
        
        # Update other birds' positions
        for i in range(1, num_birds):
            velocities[i] += compute_forces(positions[i], positions)
            velocities[i] = limit_speed(velocities[i], min_speed, max_speed)
            positions[i] += velocities[i] * time_step
        
        end = time.time()
        frame_cost = end - start
        time_cost.append(frame_cost)
        
        simulation.append(positions.copy())
        save_frame(positions, frame)
        print(f'Frame {frame} simulation time: {frame_cost:.4f}s')
    
    mean_time = np.mean(time_cost)
    print(f'Average time cost per frame: {mean_time:.4f}')
    
    create_gif()
    
    spark.stop()
