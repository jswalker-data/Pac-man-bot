import cv2
import numpy as np
import pyautogui
import time

# === Config ===
# 1) Launch the game in a window, current setup requires either full screen or the window
#    to be the full screen
# 2) Get pixel coordinates (left, top, width, height) of game area
#    Can use windows in built snipping tool

# HSV colour range for coins and pacman
# Pac-man (bright yellow)
LOWER_PACMAN = np.array([24, 200, 200])
UPPER_PACMAN = np.array([35, 255, 255])

# Normal coins (shift the pacman range more towards orange)
LOWER_COIN = np.array([18, 150, 150])
UPPER_COIN = np.array([30, 255, 255])

# Super-coins (red). We need two ranges as red wraps around Hue = 0/180
LOWER_SUPER1 = np.array([0, 150, 150])
UPPER_SUPER1 = np.array([10, 255, 255])

LOWER_SUPER2 = np.array([160, 150, 150])
UPPER_SUPER2 = np.array([180, 255, 255])

# Delay before script starting
START_DELAY = 5.0

# Delay between decision loops (s). Balance speed with CPU usage
LOOP_TIME = 0.1


def screengrab():
    """Screen grab the current game frame as a BRG numpy array"""
    screen = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)


def find_pacman(frame):
    """Locate Pac-man by thresholding the yellow and taking biggest entity"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_PACMAN, UPPER_PACMAN)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Assume Pac-man is largest entity
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return (x + w // 2, y + h // 2)


def find_coins(frame):
    """Locate the coins within an area range"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Normal coins
    m1 = cv2.inRange(hsv, LOWER_COIN, UPPER_COIN)

    # Super coins
    m2 = cv2.inRange(hsv, LOWER_SUPER1, UPPER_SUPER1)
    m3 = cv2.inRange(hsv, LOWER_SUPER2, UPPER_SUPER2)
    mask = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coins = []
    for c in contours:
        area = cv2.contourArea(c)
        if 50 < area < 300:  # guessed with these limits at the minute
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2

            # Classify
            hsv_val = hsv[cy, cx]
            is_super = (LOWER_SUPER1[0] <= hsv_val[0] <= UPPER_SUPER1[0]) or (
                LOWER_SUPER2[0] <= hsv_val[0] <= UPPER_SUPER2[0]
            )
            coins.append(((cx, cy), "super" if is_super else "normal"))
    return coins


def move_forward(src, dst):
    """Press arrow key that moves src cloaser to dst point"""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]

    if abs(dx) > abs(dy):
        pyautogui.press("right" if dx > 0 else "left")
    else:
        pyautogui.press("down" if dy > 0 else "up")


def main():
    print(f"Starting in {START_DELAY} seconds... switch to game window")
    time.sleep(START_DELAY)
    while True:
        frame = screengrab()
        pac_man = find_pacman(frame)
        if pac_man:
            coins = find_coins(frame)
            if coins:
                # Find distance to coins and pick nearest
                # Might do a more complex path finding algorithm at a later date
                # coins is type [((x, y), type), ...]
                # strip out the types
                coin_positions = [pos for pos, _ in coins]
                # then do exactly the same as before:
                dists = [
                    ((cx - pac_man[0]) ** 2 + (cy - pac_man[1]) ** 2, (cx, cy))
                    for cx, cy in coin_positions
                ]
                _, target = min(dists, key=lambda x: x[0])
                move_forward(pac_man, target)
        time.sleep(LOOP_TIME)


if __name__ == "__main__":
    main()
