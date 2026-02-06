import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import RBF

class SensorOptimizer:
    def __init__(self, area_size=(100, 100), sensor_counts=None):
        """
        Initialize the optimizer.
        area_size: (width, height) of the optimization area.
        sensor_counts: dict with keys 'temp', 'prox', 'force', 'cam' and their counts.
        """
        self.area_size = area_size
        self.sensor_counts = sensor_counts or {'temp': 2, 'prox': 2, 'force': 2, 'cam': 1}
        self.total_sensors = sum(self.sensor_counts.values())
        
        # Define the grid for evaluation (V \ S)
        x = np.linspace(0, area_size[0], 20)
        y = np.linspace(0, area_size[1], 20)
        xv, yv = np.meshgrid(x, y)
        self.grid_points = np.vstack([xv.ravel(), yv.ravel()]).T
        
        # GP Kernel for Temperature and Force
        self.kernel = RBF(length_scale=20.0)

    def decode_chromosome(self, chromosome):
        """
        Converts a flat chromosome into a structured sensor placement.
        """
        sensors = {}
        idx = 0
        for s_type, count in self.sensor_counts.items():
            sensors[s_type] = chromosome[idx:idx + count * 2].reshape(count, 2)
            idx += count * 2
        return sensors

    def calculate_gp_mi(self, sensor_locs):
        """
        Calculate Mutual Information for GP-based sensors (Temp, Force).
        MI(S; V\S) = H(S) + H(V\S) - H(V)
        For GPs, H(A) is proportional to log|K_AA|
        """
        if len(sensor_locs) == 0:
            return 0
        
        # Combine sensor locations and grid points
        all_points = np.vstack([sensor_locs, self.grid_points])
        K = self.kernel(all_points) + 1e-6 * np.eye(len(all_points))
        
        n_s = len(sensor_locs)
        K_ss = K[:n_s, :n_s]
        K_vv = K[n_s:, n_s:]
        
        # Log-determinants for entropy
        sign_ss, logdet_ss = np.linalg.slogdet(K_ss)
        sign_vv, logdet_vv = np.linalg.slogdet(K_vv)
        sign_all, logdet_all = np.linalg.slogdet(K)
        
        # MI = H(S) + H(V\S) - H(V)
        mi = 0.5 * (logdet_ss + logdet_vv - logdet_all)
        return max(0, mi)

    def calculate_proximity_coverage(self, prox_locs, radius=15.0):
        """
        Calculate information gain based on proximity coverage.
        """
        if len(prox_locs) == 0:
            return 0
        
        dists = cdist(self.grid_points, prox_locs)
        # Probability of being covered by at least one sensor
        p_covered = 1 - np.prod(np.exp(- (dists**2) / (2 * radius**2)), axis=1)
        return np.sum(p_covered)

    def calculate_camera_info(self, cam_locs, fov_deg=90, max_dist=50):
        """
        Simplified camera information based on FOV and distance.
        """
        if len(cam_locs) == 0:
            return 0
        
        # For simplicity, assume cameras point towards the center or have 360 FOV for now
        # In a real scenario, we'd include orientation in the chromosome
        dists = cdist(self.grid_points, cam_locs)
        visible = (dists < max_dist)
        # Information decreases with distance (resolution)
        info = np.sum(visible * (1 / (1 + dists/max_dist)))
        return info

    def fitness(self, chromosome):
        sensors = self.decode_chromosome(chromosome)
        
        # 1. GP-based sensors (Temp and Force)
        gp_sensors = np.vstack([sensors['temp'], sensors['force']])
        mi_gp = self.calculate_gp_mi(gp_sensors)
        
        # 2. Proximity sensors
        info_prox = self.calculate_proximity_coverage(sensors['prox'])
        
        # 3. Camera sensors
        info_cam = self.calculate_camera_info(sensors['cam'])
        
        # Weighted sum
        total_fitness = 1.0 * mi_gp + 0.1 * info_prox + 0.05 * info_cam
        
        # Penalty for sensors outside bounds
        out_of_bounds = np.sum((chromosome < 0) | (chromosome > self.area_size[0]))
        total_fitness -= out_of_bounds * 100
        
        return total_fitness

    def run_ga(self, pop_size=50, generations=100):
        # Chromosome length: 2 coordinates per sensor
        chrom_len = self.total_sensors * 2
        population = np.random.uniform(0, self.area_size[0], (pop_size, chrom_len))
        
        best_fitness = -np.inf
        best_chrom = None
        
        for gen in range(generations):
            fitnesses = np.array([self.fitness(ind) for ind in population])
            
            if np.max(fitnesses) > best_fitness:
                best_fitness = np.max(fitnesses)
                best_chrom = population[np.argmax(fitnesses)].copy()
            
            # Selection (Tournament)
            new_pop = []
            for _ in range(pop_size):
                i, j = np.random.choice(pop_size, 2)
                winner = population[i] if fitnesses[i] > fitnesses[j] else population[j]
                new_pop.append(winner)
            
            population = np.array(new_pop)
            
            # Crossover
            for i in range(0, pop_size, 2):
                if np.random.rand() < 0.8:
                    alpha = np.random.rand()
                    child1 = alpha * population[i] + (1 - alpha) * population[i+1]
                    child2 = (1 - alpha) * population[i] + alpha * population[i+1]
                    population[i], population[i+1] = child1, child2
            
            # Mutation
            mutation_rate = 0.1
            mutation_strength = 5.0
            mask = np.random.rand(*population.shape) < mutation_rate
            population += mask * np.random.normal(0, mutation_strength, population.shape)
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Best Fitness = {best_fitness:.4f}")
                
        return self.decode_chromosome(best_chrom), best_fitness

    def visualize(self, sensors, filename="sensor_placement.png"):
        plt.figure(figsize=(10, 8))
        colors = {'temp': 'red', 'prox': 'blue', 'force': 'green', 'cam': 'purple'}
        markers = {'temp': 'o', 'prox': 's', 'force': '^', 'cam': 'D'}
        
        for s_type, locs in sensors.items():
            plt.scatter(locs[:, 0], locs[:, 1], c=colors[s_type], marker=markers[s_type], 
                        label=f"{s_type.capitalize()} Sensor", s=100, edgecolors='black')
            
            # Draw proximity range
            if s_type == 'prox':
                for loc in locs:
                    circle = plt.Circle(loc, 15, color='blue', fill=False, linestyle='--', alpha=0.3)
                    plt.gca().add_patch(circle)
            
            # Draw camera range
            if s_type == 'cam':
                for loc in locs:
                    circle = plt.Circle(loc, 50, color='purple', fill=False, linestyle=':', alpha=0.2)
                    plt.gca().add_patch(circle)

        plt.xlim(0, self.area_size[0])
        plt.ylim(0, self.area_size[1])
        plt.title("Optimized Sensor Topology")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")

if __name__ == "__main__":
    optimizer = SensorOptimizer()
    best_placement, score = optimizer.run_ga(pop_size=40, generations=50)
    print("\nOptimal Placement Found:")
    for s_type, locs in best_placement.items():
        print(f"{s_type.capitalize()} Sensors: {locs.tolist()}")
    print(f"Final Fitness Score: {score:.4f}")
    optimizer.visualize(best_placement)