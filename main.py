import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import time



class Material:
    def __init__(self, color, reflection=0.0, refraction=0.0, refractive_index=1.0, emission=None):
        self.color = np.array(color, dtype=np.float64)
        self.reflection = reflection
        self.refraction = refraction
        self.refractive_index = refractive_index
        self.emission = np.array(emission, dtype=np.float64) if emission is not None else None



class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray_origin, ray_dir):
        oc = ray_origin - self.center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        t = t1 if t1 >= 0 else t2
        if t < 0:
            return None
        hit = ray_origin + t * ray_dir
        normal = normalize(hit - self.center)
        return t, hit, normal, self.material


class Plane:
    def __init__(self, point, normal, material):
        self.point = np.array(point)
        self.normal = normalize(np.array(normal))
        self.material = material

    def intersect(self, ray_origin, ray_dir):
        denom = np.dot(ray_dir, self.normal)
        if abs(denom) < 1e-6:
            return None
        t = np.dot(self.point - ray_origin, self.normal) / denom
        if t < 0:
            return None
        hit = ray_origin + t * ray_dir
        return t, hit, self.normal, self.material


class OrientedBox:
    def __init__(self, center, size, rotation_matrix, material):
        self.center = np.array(center)
        self.size = np.array(size)
        self.rotation = rotation_matrix
        self.inv_rotation = np.linalg.inv(rotation_matrix)
        self.material = material

    def intersect(self, ray_origin, ray_dir):
        local_origin = self.inv_rotation @ (ray_origin - self.center)
        local_dir = self.inv_rotation @ ray_dir

        t_min = -self.size / 2
        t_max = self.size / 2

        eps = 1e-6
        safe_local_dir = np.copy(local_dir)
        safe_local_dir[np.abs(safe_local_dir) < eps] = eps
        inv_dir = 1.0 / safe_local_dir

        t1 = (t_min - local_origin) * inv_dir
        t2 = (t_max - local_origin) * inv_dir

        t_near = np.max(np.minimum(t1, t2))
        t_far = np.min(np.maximum(t1, t2))

        if t_near > t_far or t_far < 0:
            return None

        t = t_near if t_near >= 0 else t_far
        hit_local = local_origin + t * local_dir

        normal = np.zeros(3)
        for i in range(3):
            if abs(hit_local[i] - t_min[i]) < eps:
                normal[i] = -1
                break
            elif abs(hit_local[i] - t_max[i]) < eps:
                normal[i] = 1
                break

        hit = self.rotation @ hit_local + self.center
        world_normal = normalize(self.rotation @ normal)
        return t, hit, world_normal, self.material



class PhotonMap:
    def __init__(self):
        self.positions = []
        self.powers = []
        self.normals = []
        self.kdtree = None

    def store(self, position, power, normal):
        self.positions.append(position)
        self.powers.append(power)
        self.normals.append(normal)

    def build(self):
        if self.positions:
            self.kdtree = cKDTree(self.positions)

    def estimate_radiance(self, position, normal, radius=1.0):
        if self.kdtree is None:
            return np.zeros(3)
        idxs = self.kdtree.query_ball_point(position, radius)
        total = np.zeros(3)
        count = 0
        for i in idxs:
            if np.dot(normal, self.normals[i]) > 0.5:
                total += self.powers[i]
                count += 1
        if count == 0:
            return np.zeros(3)
        return total / count



def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def reflect(I, N):
    return I - 2 * np.dot(I, N) * N


def fixed_rotation_matrix():
    theta = np.radians(30)
    psi = np.radians(35)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    return Rz @ Rx



WIDTH, HEIGHT = 800, 600
camera_pos = np.array([0, 3, -20])
MAX_DEPTH = 4
BACKGROUND = np.array([0.0, 0.0, 0.0])


light_pos = np.array([10, 30, -10])
objects = [
    Plane([0, -5, 0], [0, 1, 0], Material([0.8, 0.8, 0.8], reflection=0.2)),
    Sphere([-8, -2, 5], 3, Material([1.0, 0.2, 0.2], reflection=0.3)),
    Sphere([0, 2, 3], 2.5, Material([0.2, 0.2, 1.0], reflection=0.5, refraction=0.5, refractive_index=1.5)),
    OrientedBox([-10, 0, -5], [-7, 1, -2], fixed_rotation_matrix(), Material([0.1, 1.0, 0.1], reflection=0.2)),
    OrientedBox([7, -1, 4], [6, 6, 6], fixed_rotation_matrix(), Material([1.0, 1.0, 0.1], reflection=0.4)),
    Sphere([10, 30, -10], 3, Material([1.0, 1.0, 1.0], emission=[100, 100, 100]))
]



def backward_tracing(ray_origin, ray_dir, depth):
    if depth > MAX_DEPTH:
        return BACKGROUND

    nearest_t = np.inf
    hit_data = None

    for obj in objects:
        result = obj.intersect(ray_origin, ray_dir)
        if result:
            t, hit, normal, material = result
            if t < nearest_t:
                nearest_t = t
                hit_data = (hit, normal, material)

    if hit_data is None:
        return BACKGROUND

    hit, normal, material = hit_data

    if material.emission is not None:
        return material.emission

    to_light = normalize(light_pos - hit)
    shadow = False
    for obj in objects:
        result = obj.intersect(hit + normal * 1e-4, to_light)
        if result:
            shadow = True
            break

    ambient = 0.2
    diffuse = max(np.dot(normal, to_light), 0) if not shadow else 0
    color = ambient + 0.8 * diffuse
    local_color = material.color * color

    reflected_color = np.zeros(3)
    if material.reflection > 0:
        reflect_dir = normalize(reflect(ray_dir, normal))
        reflected_color = backward_tracing(hit + normal * 1e-4, reflect_dir, depth + 1)

    refracted_color = np.zeros(3)
    if material.refraction > 0:
        n = 1 / material.refractive_index
        cos_i = -np.dot(normal, ray_dir)
        sin2_t = n ** 2 * (1 - cos_i ** 2)
        if sin2_t <= 1:
            cos_t = np.sqrt(1 - sin2_t)
            refr_dir = normalize(n * ray_dir + (n * cos_i - cos_t) * normal)
            refracted_color = backward_tracing(hit - normal * 1e-4, refr_dir, depth + 1)

    return (
            (1 - material.reflection - material.refraction) * local_color
            + material.reflection * reflected_color
            + material.refraction * refracted_color
    )


def trace_path(ray_origin, ray_dir, depth):
    if depth >= MAX_DEPTH:
        return np.zeros(3)

    hit_data = None
    min_t = float('inf')
    for obj in objects:
        result = obj.intersect(ray_origin, ray_dir)
        if result and result[0] < min_t:
            min_t = result[0]
            hit_data = result

    if hit_data is None:
        return BACKGROUND

    t, hit, normal, material = hit_data
    if material.emission is not None:
        return material.emission


    reflected_dir = normalize(ray_dir - 2 * np.dot(ray_dir, normal) * normal)
    reflected_color = trace_path(hit + normal * 1e-4, reflected_dir, depth + 1)

    u = np.random.rand()
    v = np.random.rand()
    r = np.sqrt(u)
    theta = 2 * np.pi * v

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sqrt(max(0.0, 1.0 - u))

    w = normal
    a = np.array([1, 0, 0]) if abs(w[0]) < 0.9 else np.array([0, 1, 0])
    u = normalize(np.cross(a, w))
    v = np.cross(w, u)
    dir_world = normalize(x * u + y * v + z * w)

    diffuse_color = trace_path(hit + normal * 1e-4, dir_world, depth + 1)

    reflectivity = getattr(material, 'reflection', 0.0)
    color = (
        (1 - reflectivity) * material.color * diffuse_color +
        reflectivity * reflected_color
    )

    return color



def trace_photon(ray_origin, ray_dir, power, depth, photon_map):
    if depth >= MAX_DEPTH or np.linalg.norm(power) < 1e-3:
        return

    hit_data = None
    min_t = float('inf')
    for obj in objects:
        result = obj.intersect(ray_origin, ray_dir)
        if result and result[0] < min_t:
            min_t = result[0]
            hit_data = result

    if hit_data is None:
        return

    t, hit, normal, material = hit_data

    if material.emission is None:
        photon_map.store(hit, power, normal)

    if material.reflection > 0:
        new_dir = normalize(reflect(ray_dir, normal))
        trace_photon(hit + normal * 1e-4, new_dir, power * material.color, depth + 1, photon_map)
    else:
        r1, r2 = np.random.rand(), np.random.rand()
        phi = 2 * np.pi * r1
        theta = np.arccos(np.sqrt(1 - r2))
        sin_theta = np.sin(theta)
        local_dir = np.array([
            np.cos(phi) * sin_theta,
            np.sin(phi) * sin_theta,
            np.cos(theta)
        ])
        a = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        u = normalize(np.cross(a, normal))
        v = np.cross(normal, u)
        world_dir = normalize(local_dir[0] * u + local_dir[1] * v + local_dir[2] * normal)
        trace_photon(hit + normal * 1e-4, world_dir, power * material.color, depth + 1, photon_map)



def shade_with_reflection(hit, normal, material, depth, photon_map):
    base_color = material.color * photon_map.estimate_radiance(hit, normal, radius=0.1)

    if depth >= MAX_DEPTH or material.reflection <= 0:
        return base_color

    reflected_dir = normalize(reflect(normalize(hit - camera_pos), normal))
    hit_data = None
    min_t = float('inf')
    for obj in objects:
        result = obj.intersect(hit + reflected_dir * 1e-4, reflected_dir)
        if result and result[0] < min_t:
            min_t = result[0]
            hit_data = result

    if hit_data:
        _, hit_r, normal_r, mat_r = hit_data
        reflected_color = shade_with_reflection(hit_r, normal_r, mat_r, depth + 1, photon_map)
        return base_color * (1 - material.reflection) + reflected_color * material.reflection
    else:
        return base_color


def render_photon_mapping(filename):
    photon_map = PhotonMap()
    for _ in range(100000):  # Можешь увеличить число фотонов
        direction = normalize(np.random.randn(3))
        trace_photon(light_pos, direction, np.array([1.0, 1.0, 1.0]), 0, photon_map)

    photon_map.build()

    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()

    for y in range(HEIGHT):
        for x in range(WIDTH):
            px = (x - WIDTH / 2) / WIDTH * 2
            py = -(y - HEIGHT / 2) / HEIGHT * 2
            direction = normalize(np.array([px, py, 1]))
            color = np.zeros(3)
            min_t = float('inf')
            hit_data = None
            for obj in objects:
                result = obj.intersect(camera_pos, direction)
                if result and result[0] < min_t:
                    min_t = result[0]
                    hit_data = result

            if hit_data:
                _, hit, normal, material = hit_data
                color = shade_with_reflection(hit, normal, material, 0, photon_map)

            color = np.clip(color * 255, 0, 255).astype(np.uint8)
            pixels[x, y] = tuple(color)

    image.save(filename)


def render(tracing_function, filename):
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()
    samples = 500 if tracing_function == trace_path else 1

    for y in range(HEIGHT):
        for x in range(WIDTH):
            color = np.zeros(3)
            for _ in range(samples):
                px = (x - WIDTH / 2 + np.random.rand()) / WIDTH * 2
                py = -(y - HEIGHT / 2 + np.random.rand()) / HEIGHT * 2
                direction = normalize(np.array([px, py, 1]))
                color += tracing_function(camera_pos, direction, 0)
            color /= samples
            color = np.clip(color * 255, 0, 255).astype(np.uint8)
            pixels[x, y] = tuple(color)

    image.save(filename)



def main():
    start = time.time()
    render(backward_tracing, "render_path_tracing500.png")
    end = time.time()
    print(f"время выполнения backward_tracing: {end - start}")
    start = time.time()
    render(trace_path, "render_path_tracing500.png")
    end = time.time()
    print(f"время выполнения path_tracing_500: {end - start}")
    start = time.time()
    render_photon_mapping("newPhotonMap100000.png")
    end = time.time()
    print(f"время выполнения render_photon_mapping: {end - start}")

main()
