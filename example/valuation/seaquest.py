import torch

def in_image(img, obj):
    # img: [batch, ...] (not used, just for connectivity)
    # obj: [batch, 5]
    return obj[:, 0] # visible

def type(obj, type_onehot):
    # obj: [batch, 5] (visible, x, y, ori, type_id)
    # type_onehot: [batch, num_types]
    
    # Get type_id from obj
    if obj.shape[1] < 5:
        # print("WARNING: obj missing type_id (shape 4). Returning 0.")
        return torch.zeros_like(obj[:, 0])
        
    obj_type_id = obj[:, 4].long()
    # print(f"DEBUG: obj shape: {obj.shape}")
    # print(f"DEBUG: obj[0]: {obj[0]}")
    # print(f"DEBUG: obj_type_id unique values: {torch.unique(obj_type_id)}")
    
    # Get target type index from onehot
    target_type_id = torch.argmax(type_onehot, dim=1)
    
    # Check match
    match = (obj_type_id == target_type_id).float()
    
    # Check visibility (obj[:, 0] == 1)
    visible = obj[:, 0]
    
    return match * visible

def closeby(obj1, obj2):
    # obj: [visible, x, y, ori, type_id]
    
    x1, y1 = obj1[:, 1], obj1[:, 2]
    x2, y2 = obj2[:, 1], obj2[:, 2]
    
    dist = torch.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # Check visibility
    visible = obj1[:, 0] * obj2[:, 0]
    
    return (dist < 40.0).float() * visible

def on_left(obj1, obj2):
    x1 = obj1[:, 1]
    x2 = obj2[:, 1]
    visible = obj1[:, 0] * obj2[:, 0]
    return (x1 < x2).float() * visible

def on_right(obj1, obj2):
    x1 = obj1[:, 1]
    x2 = obj2[:, 1]
    visible = obj1[:, 0] * obj2[:, 0]
    return (x1 > x2).float() * visible

def on_top(obj1, obj2):
    y1 = obj1[:, 2]
    y2 = obj2[:, 2]
    visible = obj1[:, 0] * obj2[:, 0]
    # In Atari, y=0 is top. So on_top means y1 < y2.
    return (y1 < y2).float() * visible

def at_bottom(obj1, obj2):
    y1 = obj1[:, 2]
    y2 = obj2[:, 2]
    visible = obj1[:, 0] * obj2[:, 0]
    return (y1 > y2).float() * visible
