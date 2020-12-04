import numpy as np  # type: ignore

def print_grid(grid: np.ndarray) -> None:
    for i in range(len(grid)):
        for j in range(len(grid)):
            print("o" if grid[i, j] else ".", end=' ')
        print()

def get_norm(x):
    return np.sqrt((x**2).sum(-1))


def test_range(size, r_max) -> None:
    grid = np.zeros([size] * 2, dtype=np.bool)
    c = len(grid) // 2
    grid[c, c] = True

    gran = 10
    for i in range(1, gran):
        for j in range(1, gran):
            coords = np.array([i, j])
            coords = (coords / gran) * 2 - 1
            if 1:
                coords = np.round(coords * r_max)
                print(coords)
                norm = get_norm(coords) 
                if norm > r_max:
                    coords /= norm
                print(coords)
                print()
                coordsi = np.trunc(coords).astype(np.int32)
            else:
                norm = get_norm(coords)
                if norm > 1:
                    coords /= norm
                coords *= r_max
                coordsi = np.round(coords).astype(np.int32)
            grid[coordsi[0] + c, coordsi[1] + c] = True

    print(r_max)
    print_grid(grid)
    print()

def main() -> None:
    test_range(2 ** 3, 2 - 0.001)
    exit(0)
    for i in range(0, 3):
        test_range(2 ** 4, 1 + 2 ** i)

if __name__ == "__main__":
    main()
