import pygame, math, random

W, H = 300, 350 
FPS = 30

SPOT_W, SPOT_H = 65, 35
AISLE_WIDTH = 60
MARGIN_TOP = 25
MARGIN_SIDE = 20

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Parking game")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16, bold=True)

spots = []
grid_lines = [] 

# Limits/Walls
for row in range(6):
    y = MARGIN_TOP + row * (SPOT_H + 5)
    left_spot = pygame.Rect(MARGIN_SIDE, y, SPOT_W, SPOT_H)
    right_spot = pygame.Rect(W - MARGIN_SIDE - SPOT_W, y, SPOT_W, SPOT_H)

    spots.append(left_spot)
    spots.append(right_spot)
    grid_lines.append(((MARGIN_SIDE, y), (MARGIN_SIDE + SPOT_W, y))) # Left Top
    grid_lines.append(((W - MARGIN_SIDE - SPOT_W, y), (W - MARGIN_SIDE, y))) # Right Top

    grid_lines.append(((MARGIN_SIDE, y + SPOT_H), (MARGIN_SIDE + SPOT_W, y + SPOT_H))) # Left Bottom
    grid_lines.append(((W - MARGIN_SIDE - SPOT_W, y + SPOT_H), (W - MARGIN_SIDE, y + SPOT_H))) # Right Bottom

    grid_lines.append(((MARGIN_SIDE + SPOT_W, y), (MARGIN_SIDE + SPOT_W, y + SPOT_H)))
    grid_lines.append(((W - MARGIN_SIDE - SPOT_W, y), (W - MARGIN_SIDE - SPOT_W, y + SPOT_H)))


#Parking spots
target_idx = random.randint(0, 11)
target_rect = spots[target_idx]

obstacles = [rect for i, rect in enumerate(spots) if i != target_idx]

TARGET_POS = target_rect.center
TARGET_ANGLE = 0 

pos = [W // 2, H - 80] 
speed = 0
orientation = 90 
ang_vel = 0
steps = 0
running = True
result_msg = ""

while running:
    dt = clock.tick(FPS) / 1000
    steps += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]: speed += 6 * dt     
    elif keys[pygame.K_s]: speed -= 6 * dt   
    else: speed *= 0.8                      

    turn_speed = 140 if abs(speed) < 1 else 100

    if keys[pygame.K_a]: ang_vel = turn_speed * dt
    elif keys[pygame.K_d]: ang_vel = -turn_speed * dt
    else: ang_vel = 0

    orientation = (orientation + ang_vel) % 360
    rad = math.radians(orientation)
    pos[0] += speed * math.cos(rad)
    pos[1] -= speed * math.sin(rad)

    #Draw pickup_truck
    car_w, car_h = 42, 22
    player_rect = pygame.Rect(0, 0, car_w, car_h)
    player_rect.center = (pos[0], pos[1])

    if not (0 < pos[0] < W and 0 < pos[1] < H):
        result_msg = "CRASH! Wall."
        running = False

    hitbox = player_rect.inflate(-10, -10) 

    for obs in obstacles:
        if hitbox.colliderect(obs):
            result_msg = "CRASH! Hit a car."
            running = False

    dist_obj = math.hypot(TARGET_POS[0] - pos[0], TARGET_POS[1] - pos[1])
    ang_err = (orientation - TARGET_ANGLE + 180) % 360 - 180
    is_aligned = abs(ang_err) < 15 or abs(ang_err) > 165
    
    if dist_obj < 10 and is_aligned and abs(speed) < 0.4:
        if target_rect.collidepoint(pos[0], pos[1]):
            result_msg = "PARKED!"
            running = False

    #Draw the stage
    screen.fill((50, 55, 60)) # Asphalt color
    for line in grid_lines:
        pygame.draw.line(screen, (220, 220, 220), line[0], line[1], 2)

    #Draw cars
    for i, rect in enumerate(spots):
        if i == target_idx:
            label = font.render("PARK", True, (50, 150, 50))
            screen.blit(label, (rect.centerx - 20, rect.centery - 10))
        else:
            c_r = (i * 50) % 255
            c_g = (i * 80 + 50) % 255
            c_b = (150 + i * 20) % 255
            pygame.draw.rect(screen, (c_r, c_g, c_b), rect.inflate(-10, -10), border_radius=4)
    

    #Draw truck
    truck_surf = pygame.Surface((car_w, car_h), pygame.SRCALPHA)
    pygame.draw.rect(truck_surf, (160, 30, 30), (0, 0, 20, 22)) 
    pygame.draw.rect(truck_surf, (100, 20, 20), (2, 2, 12, 18))

    pygame.draw.rect(truck_surf, (220, 40, 40), (20, 0, 28, 22), border_radius=3)
    pygame.draw.rect(truck_surf, (180, 30, 30), (22, 2, 20, 20)) 
    pygame.draw.rect(truck_surf, (100, 200, 255), (car_w - 12, 2, 7, car_h - 4), border_radius=2)

    pygame.draw.circle(truck_surf, (255, 255, 200), (car_w-1, 4), 2)
    pygame.draw.circle(truck_surf, (255, 255, 200), (car_w-1, car_h-4), 2)

    rot_truck = pygame.transform.rotate(truck_surf, orientation)
    truck_rect = rot_truck.get_rect(center=(pos[0], pos[1]))
    screen.blit(rot_truck, truck_rect)

    ui_text = [f"Angle: {int(abs(ang_err))}", f"Speed: {speed:.1f}", f"Step: {steps}", f"Dist to spot: {int(dist_obj)}  "]
    for i, txt in enumerate(ui_text):
        screen.blit(font.render(txt, True, (255, 255, 255)), (5, H - 80 + i*18))
    pygame.display.flip()

if result_msg:
    s = pygame.Surface((W, H), pygame.SRCALPHA)
    s.fill((0,0,0,180))
    screen.blit(s, (0,0))
    msg = pygame.font.SysFont("Arial", 35, bold=True).render(result_msg, True, (255, 255, 255))
    screen.blit(msg, msg.get_rect(center=(W/2, H/2)))
    pygame.display.flip()
    pygame.time.wait(2000)

pygame.quit()