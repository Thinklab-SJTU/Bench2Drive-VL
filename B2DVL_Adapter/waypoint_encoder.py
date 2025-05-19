import math

X_TOKEN_PREFIX = 'x'
X_TOKEN_POS = 'fwd'
X_TOKEN_NEG = 'back'
X_TOKEN_ZERO = 'stay'
Y_TOKEN_PREFIX = 'y'
Y_TOKEN_POS = 'right'
Y_TOKEN_NEG = 'left'
Y_TOKEN_ZERO = 'stay'
DIR_TOKEN_PREFIX = 'dir'
DIR_TOKEN_POS = 'right'
DIR_TOKEN_NEG = 'left'
DIR_TOKEN_ZERO = 'fwd'
DIR_TOKEN_BACK = 'back'
SPD_TOKEN_PREFIX = 'spd'

ALL_TOKENS_FILE = './all_tokens.txt'

def float_to_token(val):
    abs_val = abs(val)
    
    if abs_val >= 15 + 15/16:
        return 'ff'
    
    token = round(abs_val * 16)
    return f'{token:02x}'

def generate_motion_tokens(points):
    motion_tokens = []
    
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        
        x_token = float_to_token(x)
        if x_token == '00':
            x_token = f"<{X_TOKEN_PREFIX}_{X_TOKEN_ZERO}_00>"
        elif x > 0:
            x_token = f"<{X_TOKEN_PREFIX}_{X_TOKEN_POS}_{x_token}>"
        else:
            x_token = f"<{X_TOKEN_PREFIX}_{X_TOKEN_NEG}_{x_token}>"
        
        y_token = float_to_token(y)
        if y_token == '00':
            y_token = f"<{Y_TOKEN_PREFIX}_{Y_TOKEN_ZERO}_00>"
        elif y > 0:
            y_token = f"<{Y_TOKEN_PREFIX}_{Y_TOKEN_POS}_{y_token}>"
        else:
            y_token = f"<{Y_TOKEN_PREFIX}_{Y_TOKEN_NEG}_{y_token}>"
        
        motion_tokens.append(x_token)
        motion_tokens.append(y_token)

    return motion_tokens

def angle_to_token(angle):
    if angle < -180:
        angle += 360
    elif angle >= 180:
        angle -= 360
    
    abs_angle = abs(angle)
    
    if abs_angle <= 180:
        token = round(abs_angle / 180 * 64)
    else:
        token = 64
    
    if token == 0:
        return f"<{DIR_TOKEN_PREFIX}_{DIR_TOKEN_ZERO}_00>"
    elif token == 64:
        return f"<{DIR_TOKEN_PREFIX}_{DIR_TOKEN_BACK}_40>"
    
    direction = DIR_TOKEN_POS if angle > 0 else DIR_TOKEN_NEG
    
    return f"<{DIR_TOKEN_PREFIX}_{direction}_{token:02x}>"

def distance_to_token(distance):
    if distance >= 15 + 15/16:
        return f"<{SPD_TOKEN_PREFIX}_ff>"
    
    token = round(distance * 16)
    return f"<{SPD_TOKEN_PREFIX}_{token:02x}>"

def generate_motion_and_direction_tokens(points):
    motion_tokens = []
    
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        
        angle = math.degrees(math.atan2(y, x))
        
        distance = math.sqrt(x**2 + y**2)
        
        dir_token = angle_to_token(angle)
        spd_token = distance_to_token(distance)
        
        if "00" in spd_token:
            dir_token = f"<{DIR_TOKEN_PREFIX}_{DIR_TOKEN_ZERO}_00>"
        
        motion_tokens.append(dir_token)
        motion_tokens.append(spd_token)
    
    return motion_tokens

def list_all_tokens():
    token_list = []
    token_list.append(f"<{X_TOKEN_PREFIX}_{X_TOKEN_ZERO}_00>")
    for i in range (1, 256):
        token_list.append(f"<{X_TOKEN_PREFIX}_{X_TOKEN_POS}_{i:02x}>")
    for i in range (1, 256):
        token_list.append(f"<{X_TOKEN_PREFIX}_{X_TOKEN_NEG}_{i:02x}>")
    token_list.append(f"<{Y_TOKEN_PREFIX}_{Y_TOKEN_ZERO}_00>")
    for i in range (1, 256):
        token_list.append(f"<{Y_TOKEN_PREFIX}_{Y_TOKEN_POS}_{i:02x}>")
    for i in range (1, 256):
        token_list.append(f"<{Y_TOKEN_PREFIX}_{Y_TOKEN_NEG}_{i:02x}>")
    token_list.append(f"<{DIR_TOKEN_PREFIX}_{DIR_TOKEN_ZERO}_00>")
    token_list.append(f"<{DIR_TOKEN_PREFIX}_{DIR_TOKEN_BACK}_40>")
    for i in range (1, 64):
        token_list.append(f"<{DIR_TOKEN_PREFIX}_{DIR_TOKEN_POS}_{i:02x}>")
    for i in range (1, 64):
        token_list.append(f"<{DIR_TOKEN_PREFIX}_{DIR_TOKEN_NEG}_{i:02x}>")
    for i in range (0, 256):
        token_list.append(f"<{SPD_TOKEN_PREFIX}_{i:02x}>")
    return token_list

def print_all_tokens(sep=','):
    token_str = sep.join(list_all_tokens())
    with open(ALL_TOKENS_FILE, 'w') as file:
        file.write(token_str)

if __name__ == "__main__":
    print_all_tokens(',')