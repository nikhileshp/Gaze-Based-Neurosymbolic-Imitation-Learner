from .utils import find_objects, match_objects
from .game_objects import GameObject, NoObject

objects_colors = {"player": [[187, 187, 53], [236, 236, 236]], "diver": [66, 72, 200], "background_water": [0, 28, 136],
                  "player_score": [210, 210, 64], "oxygen_bar": [214, 214, 214], "lives": [210, 210, 64],
                  "logo": [66, 72, 200], "player_missile": [187, 187, 53], "oxygen_bar_depleted": [163, 57, 21],
                  "oxygen_logo": [0, 0, 0], "collected_diver": [24, 26, 167], "enemy_missile": [66, 72, 200],
                  "submarine": [170, 170, 170]}

enemy_colors = {"green": [92, 186, 92], "orange": [198, 108, 58], "yellow": [160, 171, 79], "lightgreen": [72, 160, 72],
                "pink": [198, 89, 179]}


class Player(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 187, 187, 53


class Diver(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 66, 72, 200


class Shark(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 92, 186, 92


class Submarine(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 170, 170, 170


class PlayerMissile(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 187, 187, 53


class OxygenBar(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 214, 214, 214


class OxygenBarDepleted(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 163, 57, 21


class OxygenBarLogo(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 0, 0, 0


class PlayerScore(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 210, 210, 64


class Lives(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 210, 210, 64


class CollectedDiver(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 24, 26, 167


class EnemyMissile(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 66, 72, 200


class Surface(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 0, 0, 0


def _detect_objects(objects, obs, hud=False):
    player = []
    for color in objects_colors["player"]:
        player.extend(find_objects(obs, color, closing_dist=8))

    for p in player:
        if p[1] > 30 and p[3] > 6:
            objects[0].xywh = p
            
            # Determine orientation based on pixel density
            # User heuristic: "number of pixels of the player greater in the right half and left half"
            # Player colors
            p_cols = objects_colors["player"]
            
            # Crop player image
            x, y, w, h = p
            player_img = obs[y:y+h, x:x+w]
            
            # Split into left/right
            mid_w = w // 2
            left_half = player_img[:, :mid_w]
            right_half = player_img[:, mid_w:]
            
            # Count pixels matching player colors
            # Simple approach: sum of boolean masks for each color
            # Or simplified: non-background count?
            # Let's match specific player colors for accuracy
            left_count = 0
            right_count = 0
            
            import numpy as np
            for col in p_cols:
                # col is [R, G, B] list
                c = np.array(col)
                # Check match (e.g. within tolerance or exact)
                # Ocatari usually exact match for these constants
                left_count += np.sum(np.all(left_half == c, axis=-1))
                right_count += np.sum(np.all(right_half == c, axis=-1))
            
            # Define Orientation class locally or stub
            class Orientation:
                def __init__(self, val):
                    self.value = val
            
            if right_count > left_count:
                objects[0].orientation = Orientation(4) # Right -> 4 (from valuation.py check)
            else:
                objects[0].orientation = Orientation(12) # Left -> 12
                
            player.remove(p)
            break
    if player:
        for p in player:
            if p[1] > 30 and p[3] == 1 and p[2] == 8:
                if type(objects[34]) is NoObject:
                    objects[34] = PlayerMissile(*p)
                else:
                    objects[34].xywh = p
                break
    else:
        objects[34] = NoObject()

    divers_and_missiles = find_objects(
        obs, objects_colors["diver"], closing_dist=1)
    divers = []
    missiles = []
    for dm in divers_and_missiles:
        if dm[1] < 190 and dm[2] > 2 and dm[3] > 3:
            divers.append(dm)
        elif dm[1] < 190 and dm[2] > 2:
            missiles.append(dm)

    match_objects(objects, divers[:4], 25, 4, Diver)
    match_objects(objects, missiles[:4], 29, 4, EnemyMissile)

    shark = []
    for enemyColor in enemy_colors.values():
        shark.extend(find_objects(obs, enemyColor, min_distance=1))

    match_objects(objects, shark[:12], 1, 12, Shark)

    submarine_all = find_objects(obs, objects_colors["submarine"], min_distance=1, size=(10,12), tol_s=2)
    
    # Detect surface submarines separately: same gray color but in y=40-53 band.
    # They are smaller (approx 8x7) so we use relaxed size constraints.
    import numpy as np
    surface_band = obs.copy()
    surface_band[:40, :, :] = 0   # blank out everything above band
    surface_band[54:, :, :] = 0   # blank out everything below band
    surface_subs_raw = find_objects(surface_band, objects_colors["submarine"], min_distance=1, size=(4,3), tol_s=8)
    
    # Assign surface sub to slot 33 BEFORE match_objects so it doesn't steal a Submarine slot
    if surface_subs_raw:
        s = surface_subs_raw[0]
        if type(objects[33]) is NoObject:
            objects[33] = Submarine(*s)
        else:
            objects[33].xywh = s
    else:
        objects[33] = NoObject()

    # Map underwater submarines (full-size gray, not in surface band) to slots 13-24
    match_objects(objects, submarine_all[:12], 13, 12, Submarine)
    
    oxygen_bar = find_objects(
        obs, objects_colors["oxygen_bar"], min_distance=1)
    if oxygen_bar:
        if type(objects[35]) is NoObject:
            objects[35] = OxygenBar(*oxygen_bar[0])
        else:
            objects[35].xywh = oxygen_bar[0]
    else:
        objects[35] = NoObject()

    coll_diver = find_objects(obs, objects_colors["collected_diver"])

    if coll_diver:
        x, y, w, h = coll_diver[0]
        for i in range(6):
            if i < w/8:
                if type(objects[36+i]) is NoObject:
                    objects[36+i] = CollectedDiver(x+8*i, y, 8, h)
            else:
                objects[36+i] = NoObject()
    else:
        for i in range(6):
            if type(objects[36+i]) != NoObject:
                objects[36+i] = NoObject()

    # Detect Surface - Black line between y=45 and y=55
    # Create a synthetic surface object spanning the width of the screen
    import numpy as np
    surface_region = obs[45:56, :, :]  # y-range 45-55, full width
    black_color = np.array([0, 0, 0])
    black_pixels = np.all(surface_region == black_color, axis=-1)

    if np.any(black_pixels):
        # Surface exists - create full-width object at y=55 (where the black line is)
        if type(objects[42]) is NoObject:
            objects[42] = Surface(0, 55, 160, 1)  # x=0, y=55, w=160 (full screen), h=1
        else:
            objects[42].xywh = (0, 55, 160, 1)
    else:
        objects[42] = NoObject()

    if hud:
        score = find_objects(
            obs, objects_colors["player_score"], maxy=17, min_distance=1, closing_dist=5)
        objects[-4].xywh = score[0]

        lives = find_objects(
            obs, objects_colors["player_score"], miny=22, maxy=30, min_distance=1, closing_dist=10)
        objects[-3].xywh = lives[0]

        oxygen_bar_depl = find_objects(
            obs, objects_colors["oxygen_bar_depleted"], min_distance=1)
        if oxygen_bar_depl:
            if type(objects[-2]) is NoObject:
                objects[-2] = OxygenBarDepleted(*oxygen_bar_depl[0])
            else:
                objects[-2].xywh = oxygen_bar_depl[0]
        else:
            objects[-2] = NoObject()

        # oxygen_logo = find_objects(obs, objects_colors["oxygen_logo"], min_distance=1)
        # for ox_logo in oxygen_logo:
        #     if ox_logo[0] > 0:
        #         objects.append(OxygenBarLogo(*ox_logo))
