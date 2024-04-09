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

    def square_boundary(self, padding=None):
        """
        Generates a mask for a square boundary with a specified padding from the edges.
        """
        if padding is None:
            padding = np.random.randint(1, 20)
        mask = np.ones((self.nx, self.ny))
        mask[padding:-padding, padding:-padding] = 0
        return mask.astype(bool)

    def elliptical_boundary(self):
        """
        Generate a numpy array representing an ellipse.
        Inside the ellipse is 0, outside is 1.
        """
        # Create a meshgrid of coordinates x, y
        # x, y = np.ogrid[:self.nx, :self.ny]
        x = np.linspace(0, self.nx, self.nx)
        y = np.linspace(0, self.ny, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        if self.nx > self.ny:
            max_ax = self.nx / 2
            min_ax = self.ny / 2
        else:
            max_ax = self.ny / 2
            min_ax = self.nx / 2

        # Center of the ellipse
        maj_cent = np.random.uniform(max_ax * 0.9, (self.nx - max_ax) * 1.1)
        min_cent = np.random.uniform(min_ax * 0.9, (self.ny - min_ax) * 1.1)

        # Semi-major and semi-minor axes (ensure a > b for convention)
        a = np.random.uniform(max_ax / 1.33, max_ax*1.1)
        b = np.random.uniform(min_ax / 1.33, min_ax*1.1)

        # Orientation angle in degrees
        theta = np.random.uniform(-30, 30)

        # Calculate the ellipse equation
        theta_rad = np.deg2rad(theta)  # Convert theta to radians for np.cos and np.sin
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        if self.nx > self.ny:
            term1 = ((X - maj_cent) * cos_theta + (Y - min_cent) * sin_theta) ** 2 / a ** 2
            term2 = ((X - maj_cent) * sin_theta - (Y - min_cent) * cos_theta) ** 2 / b ** 2
        else:
            term1 = ((X - maj_cent) * cos_theta - (Y - min_cent) * sin_theta) ** 2 / b ** 2
            term2 = ((X - maj_cent) * sin_theta + (Y - min_cent) * cos_theta) ** 2 / a ** 2

        # Inside the ellipse <= 1, so we invert the condition to set inside to 0, outside to 1
        ellipse = 1 - (term1 + term2 <= 1).astype(int)  # .T
        # Set boundary to 1 always
        ellipse[0, :] = 1
        ellipse[-1, :] = 1
        ellipse[:, 0] = 1
        ellipse[:, -1] = 1
        return ellipse.astype(bool)

    def random_polygon_boundary(self, n_points=20):
        """
        Generates a mask for a random polygon boundary defined by a convex hull of 'n_points' points.
        """
        rand_points = np.random.rand(n_points, 2) * [self.nx, self.ny]
        hull = ConvexHull(rand_points)

        # Create a path from the convex hull
        path = mpath.Path(rand_points[hull.vertices])

        # Rasterize the path to a grid
        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        points = np.vstack((X.ravel(), Y.ravel())).T
        grid = path.contains_points(points)
        mask = grid.reshape((self.nx, self.ny)).astype(int)

        # print(mask.shape)
        # plt.imshow(mask.T, origin='lower')
        # plt.scatter(rand_points[:, 0], rand_points[:, 1], c='r')
        # plt.show()
        # exit(5)
        # Invert mask if necessary so the outside of the polygon is 1, inside is 0
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
