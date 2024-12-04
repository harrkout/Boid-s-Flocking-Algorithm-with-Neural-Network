
import pygame
import random
import math
import numpy as np
import csv

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Boid settings
NUM_BOIDS = 50
MAX_SPEED = 4
MAX_FORCE = 0.1
BOID_RADIUS = 5
PERCEPTION_RADIUS = 50

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 150, 255)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)

# Flags for features
mouse_attraction = False
show_metrics = False
enable_predator = True

# Saving dataset to CSV
def save_data_to_csv(inputs, outputs, filename="boid_data.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow([f"input_{i}" for i in range(inputs.shape[1])] + [f"output_{i}" for i in range(outputs.shape[1])])
        # Write input-output pairs
        for input_row, output_row in zip(inputs, outputs):
            writer.writerow(list(input_row) + list(output_row))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.hidden = self.sigmoid(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def train(self, x, y, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            output = self.forward(x)
            error = y - output
            d_output = error * self.sigmoid_derivative(output)
            d_hidden = d_output.dot(self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden)

            self.weights_hidden_output += self.hidden.T.dot(d_output) * learning_rate
            self.weights_input_hidden += x.T.dot(d_hidden) * learning_rate
            self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
            self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

class Boid:
    def __init__(self, x, y):
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        self.velocity.scale_to_length(MAX_SPEED)
        self.acceleration = pygame.Vector2(0, 0)

    def edges(self):
        if self.position.x > WIDTH:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = WIDTH
        if self.position.y > HEIGHT:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = HEIGHT

    def align(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        for other in boids:
            if other != self and self.position.distance_to(other.position) < PERCEPTION_RADIUS:
                steering += other.velocity
                total += 1
        if total > 0:
            steering /= total
            steering.scale_to_length(MAX_SPEED)
            steering -= self.velocity
            steering = self.limit_force(steering)
        return steering

    def cohesion(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        for other in boids:
            if other != self and self.position.distance_to(other.position) < PERCEPTION_RADIUS:
                steering += other.position
                total += 1
        if total > 0:
            steering /= total
            steering -= self.position
            steering.scale_to_length(MAX_SPEED)
            steering -= self.velocity
            steering = self.limit_force(steering)
        return steering

    def separation(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        for other in boids:
            distance = self.position.distance_to(other.position)
            if other != self and distance < PERCEPTION_RADIUS / 2:
                diff = self.position - other.position
                diff /= distance
                steering += diff
                total += 1
        if total > 0:
            steering /= total
            steering.scale_to_length(MAX_SPEED)
            steering -= self.velocity
            steering = self.limit_force(steering)
        return steering

    def seek_mouse(self, mouse_pos, repel=False):
        target = pygame.Vector2(mouse_pos)
        direction = target - self.position if not repel else self.position - target
        if direction.length() > 0:
            direction.scale_to_length(MAX_SPEED)
            steering = direction - self.velocity
            steering = self.limit_force(steering)
            self.acceleration += steering

    def avoid_predator(self, predator_pos):
        if self.position.distance_to(predator_pos) < PERCEPTION_RADIUS:
            diff = self.position - predator_pos
            diff.scale_to_length(MAX_SPEED)
            steering = diff - self.velocity
            self.acceleration += self.limit_force(steering)

    def limit_force(self, force):
        if force.length() > MAX_FORCE:
            force.scale_to_length(MAX_FORCE)
        return force

    def flock(self, boids, nn):
        neighbors = [other for other in boids if other != self and self.position.distance_to(other.position) < PERCEPTION_RADIUS]
        if neighbors:
            align_force = self.align(neighbors)
            cohesion_force = self.cohesion(neighbors)
            separation_force = self.separation(neighbors)

            input_features = np.array([
                self.position.x, self.position.y,
                self.velocity.x, self.velocity.y,
                align_force.x, align_force.y,
                cohesion_force.x, cohesion_force.y,
                separation_force.x, separation_force.y,
            ]).reshape(1, -1)

            steering = nn.forward(input_features)
            self.acceleration += pygame.Vector2(steering[0, 0], steering[0, 1])

    def update(self):
        self.velocity += self.acceleration
        if self.velocity.length() > MAX_SPEED:
            self.velocity.scale_to_length(MAX_SPEED)
        self.position += self.velocity
        self.acceleration *= 0

    def draw(self, screen):
        angle = math.atan2(self.velocity.y, self.velocity.x)
        point1 = self.position + pygame.Vector2(math.cos(angle), math.sin(angle)) * BOID_RADIUS
        point2 = self.position + pygame.Vector2(math.cos(angle + 2.5), math.sin(angle + 2.5)) * BOID_RADIUS
        point3 = self.position + pygame.Vector2(math.cos(angle - 2.5), math.sin(angle - 2.5)) * BOID_RADIUS
        pygame.draw.polygon(screen, BLUE, [point1, point2, point3])


# Calculate metrics
def calculate_metrics(boids):
    total_speed = sum(boid.velocity.length() for boid in boids)
    average_speed = total_speed / len(boids)

    center_of_mass = pygame.Vector2(
        sum(boid.position.x for boid in boids) / len(boids),
        sum(boid.position.y for boid in boids) / len(boids),
    )
    cohesion_measure = sum(boid.position.distance_to(center_of_mass) for boid in boids) / len(boids)

    return average_speed, cohesion_measure


def display_menu(screen, font):
    """Display the menu for toggling features."""
    menu_texts = [
        "Press 1: Toggle Mouse Attraction",
        "Press 2: Toggle Predator",
        "Press 3: Show Metrics",
        "Press Esc: Quit",
    ]
    for i, text in enumerate(menu_texts):
        menu_surface = font.render(text, True, WHITE)
        screen.blit(menu_surface, (10, 10 + i * 20))


# Display metrics
def display_metrics(screen, font, average_speed, cohesion_measure):
    metrics_texts = [
        f"Average Speed: {average_speed:.2f}",
        f"Cohesion Measure: {cohesion_measure:.2f}",
    ]
    for i, text in enumerate(metrics_texts):
        metrics_surface = font.render(text, True, YELLOW)
        screen.blit(metrics_surface, (10, HEIGHT - 40 + i * 20))



def main():
    global mouse_attraction, show_metrics, enable_predator

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    input_size = 10  # 10 input features (position, velocity, alignment, cohesion, separation)
    hidden_size = 5
    output_size = 2  # 2 outputs (x, y direction of the steering force)

    # Create the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Generate boids
    boids = [Boid(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(NUM_BOIDS)]
    predator_pos = pygame.Vector2(random.randint(0, WIDTH), random.randint(0, HEIGHT))

    # Training the neural network
    print("Training the neural network...")
    inputs = []
    outputs = []

    # Generate synthetic training data
    for boid in boids:
        align_force = boid.align(boids)
        cohesion_force = boid.cohesion(boids)
        separation_force = boid.separation(boids)

        input_features = [
            boid.position.x, boid.position.y,
            boid.velocity.x, boid.velocity.y,
            align_force.x, align_force.y,
            cohesion_force.x, cohesion_force.y,
            separation_force.x, separation_force.y,
        ]
        inputs.append(input_features)

        # Example desired output: steering towards the screen center
        screen_center = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
        desired_direction = (screen_center - boid.position).normalize() * MAX_SPEED
        outputs.append([desired_direction.x, desired_direction.y])

    # Convert to NumPy arrays
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # Train the neural network
    nn.train(inputs, outputs, epochs=1000, learning_rate=0.1)
    print("Training completed!")

    # Main loop
    running = True
    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    mouse_attraction = not mouse_attraction
                if event.key == pygame.K_2:
                    enable_predator = not enable_predator
                if event.key == pygame.K_3:
                    show_metrics = not show_metrics
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Update and display boids
        for boid in boids:
            boid.edges()

            if mouse_attraction:
                boid.seek_mouse(pygame.mouse.get_pos())
            if enable_predator:
                boid.avoid_predator(predator_pos)
            boid.flock(boids, nn)
            boid.update()
            boid.draw(screen)

        # Update predator position to follow the mouse cursor
        if enable_predator:
            predator_pos.update(pygame.mouse.get_pos())

        # Draw predator
        if enable_predator:
            pygame.draw.circle(screen, RED, (int(predator_pos.x), int(predator_pos.y)), 10)

        # Show menu and metrics
        display_menu(screen, font)
        if show_metrics:
            average_speed, cohesion_measure = calculate_metrics(boids)
            display_metrics(screen, font, average_speed, cohesion_measure)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

