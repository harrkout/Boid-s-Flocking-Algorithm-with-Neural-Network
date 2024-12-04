# Boid Simulation with Neural Network Integration

## Overview
This project simulates the behavior of boids (artificial agents) based on flocking algorithms inspired by real-world bird flocks and fish schools. It incorporates neural networks to control boid steering behavior and includes various interactive features such as mouse attraction, predator avoidance, and real-time metrics display.



![img](/img.png)



## Features
- **Flocking Behavior**: Implements alignment, cohesion, and separation rules.

- **Neural Network Integration**: A simple feedforward neural network processes environmental inputs and adjusts boid steering.

- **Interactive Features**:
  - Toggle mouse attraction.
  - Enable or disable a predator to influence boid behavior.
  - Display real-time metrics.
  
- **Metrics Calculation**:
  - Average speed.
  - Cohesion measure based on the boids' center of mass.
  
- **Data Export**: Save boid simulation data (inputs and outputs) to a CSV file.

  

## Usage
1. Use the following keys to interact with the simulation:
   - **1**: Toggle mouse attraction.
   - **2**: Enable/disable the predator.
   - **3**: Show/hide real-time metrics.
   - **ESC**: Quit the simulation.

## Controls and Features
### Mouse Attraction
- Toggle the boids' attraction or repulsion to the mouse cursor.

### Predator Avoidance
- When enabled, a predator moves on the screen, and boids attempt to avoid it if within their perception radius.

### Metrics Display
- Real-time metrics such as average speed and cohesion measure are shown on the screen when enabled.

### Data Saving
- Save simulation data to a CSV file for further analysis or training other machine learning models:
  ```python
  save_data_to_csv(inputs, outputs, filename="boid_data.csv")
  ```

## Code Structure
### Main Components
1. **Boid Class**: Represents an individual boid, implementing flocking behavior and interaction logic.
   - **Methods**:
     - `align`: Steer towards the average velocity of neighbors.
     - `cohesion`: Steer towards the center of mass of neighbors.
     - `separation`: Steer away from close neighbors to avoid crowding.
     - `flock`: Combine alignment, cohesion, and separation forces.
     - `update`: Update position and velocity based on acceleration.
     - `draw`: Render the boid on the screen.

2. **Neural Network Class**: Implements a simple feedforward neural network.
   - **Methods**:
     - `forward`: Perform a forward pass.
     - `train`: Train the network using gradient descent.

3. **Metrics**:
   - `calculate_metrics`: Computes average speed and cohesion measure.
   - `display_metrics`: Renders metrics on the screen.

4. **Interaction**:
   - `display_menu`: Displays instructions for toggling features.

### Simulation Flow
1. Initialize boids and neural network.
2. Process boid behaviors (alignment, cohesion, separation).
3. Integrate neural network outputs for additional steering adjustments.
4. Update boid positions.
5. Render the simulation.
6. Handle user inputs for toggling features.

## Customization
- **Boid Settings**:
  Adjust parameters like number of boids, maximum speed, and perception radius at the top of the script.
  ```python
  NUM_BOIDS = 50
  MAX_SPEED = 4
  PERCEPTION_RADIUS = 50
  ```
- **Neural Network**:
  Modify the architecture by adjusting input, hidden, and output layer sizes.
  ```python
  nn = NeuralNetwork(input_size=10, hidden_size=16, output_size=2)
  ```
- **Dataset Saving**:
  Use `save_data_to_csv` to export simulation data for further analysis.

