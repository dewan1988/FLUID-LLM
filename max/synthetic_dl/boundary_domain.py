import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib.path as mpath


class BoundaryConditionGenerator:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

    def random_boundary(self):
        """
        Randomly selects a type of boundary to return
        """
        rnd = np.random.randint(3)
        if rnd == 0:
            return self.square_boundary()
        elif rnd == 1:
            return self.elliptical_boundary()
        elif rnd == 2:
            return self.random_polygon_boundary()
        else:
            raise ValueError("Off by 1")

    def square_boundary(self, padding=5):
        """
        Generates a mask for a square boundary with a specified padding from the edges.
        """
        mask = np.ones((self.nx, self.ny))
        mask[padding:-padding, padding:-padding] = 0
        return mask.astype(bool)

    def elliptical_boundary(self):
        """
        Generate a numpy array representing an ellipse.
        Inside the ellipse is 0, outside is 1.
        """
        # Create a meshgrid of coordinates x, y
        y, x = np.ogrid[:self.ny, :self.nx]

        margin = max(self.nx, self.ny) / 2
        # Center of the ellipse
        x0 = np.random.uniform(margin * 0.9, (self.nx - margin) * 1.1)
        y0 = np.random.uniform(margin * 0.9, (self.ny - margin) * 1.1)

        # Semi-major and semi-minor axes (ensure a > b for convention)
        a = np.random.uniform(margin / 1.5, margin)
        b = np.random.uniform(margin / 2, a)

        # Orientation angle in degrees
        theta = np.random.uniform(0, 360)

        # Calculate the ellipse equation
        theta_rad = np.deg2rad(theta)  # Convert theta to radians for np.cos and np.sin
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        term1 = ((x - x0) * cos_theta + (y - y0) * sin_theta) ** 2 / a ** 2
        term2 = ((x - x0) * sin_theta - (y - y0) * cos_theta) ** 2 / b ** 2

        # Inside the ellipse <= 1, so we invert the condition to set inside to 0, outside to 1
        ellipse = 1 - (term1 + term2 <= 1).astype(int)

        return ellipse.astype(bool)

    def random_polygon_boundary(self, n_points=15):
        """
        Generates a mask for a random polygon boundary defined by a convex hull of 'n_points' points.
        """
        points = np.random.rand(n_points, 2) * [self.nx, self.ny]
        hull = ConvexHull(points)

        # Create a path from the convex hull
        path = mpath.Path(points[hull.vertices])

        # Rasterize the path to a grid
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        points = np.vstack((X.ravel(), Y.ravel())).T
        grid = path.contains_points(points)
        mask = grid.reshape((self.nx, self.ny)).astype(int)

        # Invert mask if necessary so the inside of the polygon is 1, outside is 0
        return (1 - mask).astype(bool)


if __name__ == "__main__":
    # Example usage
    nx, ny = 100, 100
    generator = BoundaryConditionGenerator(nx, ny)

    # Generate boundary condition masks
    square_mask = generator.square_boundary(padding=10)
    elliptical_mask = generator.elliptical_boundary()
    irregular_mask = generator.random_polygon_boundary()

    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(square_mask, origin='lower')
    ax[0].set_title("Square Boundary")
    ax[1].imshow(elliptical_mask, origin='lower')
    ax[1].set_title("Elliptical Boundary")
    ax[2].imshow(irregular_mask, origin='lower')
    ax[2].set_title("Random Irregular Boundary")
    plt.show()
