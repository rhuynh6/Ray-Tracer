'''
RAY TRACER 
code by Ruby Huynh
Features added:
-Reflection
-Shadows
-Ground plane
'''

import numpy as np
import matplotlib.pyplot as plotting

#4:3 aspect ratio
WIDTH = 480
HEIGHT = 360

MAX_REFLECT=5

class ball:
    def __init__(self, x, y, z, r, g, b,        # position, color
                 rad = 0.3, refl=0.25, kambient= 0.05, kdiffuse= 1, kspec=50):
        self.position=np.array([x, y, z])
        self.color = np.array([r, g, b],  dtype = float)
        self.reflection=refl
        self.radius = rad
        self.ka = kambient
        self.kd = kdiffuse
        self.ks = kspec

class pos:
    def __init__(self, x, y, z):
        self.position = np.array([x, y, z], dtype = float)

class color:
    def __init__(self, r, g, b):
        self.col = np.array([r, g, b],  dtype = float)

class light:
    def __init__(self, x, y, z, r, g, b,
                kambient=.05, kdiffuse=1., kspec=70):
        self.position=np.array([x, y, z])
        self.color=np.array([r, g, b],  dtype = float)
        self.ka = kambient
        self.kd = kdiffuse
        self.ks = kspec

class plane:
    def __init__(self, x, y, z, n1, n2, n3, r=1, g=0.945, b=0.741,
                refl=0.25, kambient= 0.05, kdiffuse= 0.75, kspec=0.5):
        self.position=np.array([x, y, z])
        self.normal=np.array([n1, n2, n3])
        self.color=np.array([r, g, b],  dtype = float)
        self.reflection=refl
        self.ka = kambient
        self.kd = kdiffuse
        self.ks = kspec

def normalize(x):
    x = x/(np.linalg.norm(x))
    return x

def normal(object, intersection):
    if type(object).__name__ == 'ball':
        surf_normal = normalize (intersection-object.position)
    elif type(object).__name__ == 'plane':
        surf_normal=object.normal
    return surf_normal

def ball_distance(radius, position, source, direction):
    sp_diff=source-position
    a=np.dot(direction, direction)
    b=2*np.dot(direction, sp_diff)
    c=np.dot(sp_diff, sp_diff)-radius*radius

    det=b*b-4*a*c
    if det > 0:
        detSqrt=np.sqrt(det)
        if b<0:
            t=(-b-detSqrt)/2.0
        else:
            t=(-b+detSqrt)/2.0
    
        t0=t/a
        t1=c/t

        t0=min(t0,t1) #get the lesser 
        t1=max(t0,t1) #get the greater
    
        if t1>=0:
            return t1 if t0<0 else t0
    return np.inf

def plane_distance(position, normal, source, direction):
    denom = np.dot(direction, normal)
    if np.abs(denom) < 1e-6:
        return np.inf
    distance = np.dot(position - source, normal) / denom
    if distance < 0:
        return np.inf
    return distance

def distance(source, direction, object):
    if type(object).__name__ == 'ball':
        return ball_distance(object.radius, object.position, source, direction)
    if type(object).__name__ == 'plane':
        return plane_distance(object.position, object.normal, source, direction)

def ray_trace(source, direction):
    ray_destination=np.inf
    for i, object in enumerate(scene):
        ray_hit=distance(source, direction, object)
        if ray_hit < ray_destination:
            ray_destination=ray_hit
            hit_index=i
    
    #Primary ray doesn't hit anything
    if ray_destination==np.inf:
        return

    object=scene[hit_index]
    intersect_point=source+ray_destination*direction
    toLight=normalize(scene_light.position-intersect_point)
    toSource=normalize(source-intersect_point)
    surf_normal=normal(object, intersect_point)

    shadow_values = []
    for k, obj_sh in enumerate(scene):
        # Skip the object that was hit
        if k == hit_index:
            continue

        # Calculate the intersection point slightly offset along the surface normal
        offset_intersect_point = intersect_point + surf_normal * 0.0001

        # Calculate the distance from the offset intersection point to the light source
        distance_to_light = distance(offset_intersect_point, toLight, obj_sh)

        # Append the calculated shadow value to the list
        shadow_values.append(distance_to_light)
    
    if shadow_values and min(shadow_values)<np.inf:
        return
    
    Ia=object.ka
    Id=object.kd*max(np.dot(surf_normal, toLight), 0) * object.color
    Is= object.ks* max(np.dot(surf_normal, normalize(toLight + toSource)), 0) ** scene_light.ks * scene_light.color
    col_ray=Ia+Id+Is
    return object, intersect_point, surf_normal, col_ray

eye_pos=np.array([0., 0.35, -1.])
eye_direction=np.array([0., 0., 0.])
scene_light=light(7., 4., -10., 0.63, 1., 0.349)
curr_color=np.zeros(3)
img = np.zeros((HEIGHT, WIDTH, 3))

scene=[ball(-0.75, .5, 1., .278, .573, .710), 
       ball(1.75, .2, 2.25, .302, .419, .412), 
       ball(1, .1, 3.5, .529, .702, .471),
       ball(2.75, .1, 2.25, .278, .573, .710),
       ball(-.5, 2.0, 2.0, .302, .419, .412),
       ball(-2.0, 3.7, 1.25, .529, .702, .471),
       ball(0., .6, 0.6, .278, .573, .710),
       ball(0.1, 2.0, 0.9, .302, .419, .412),
       ball(-0.1, 2.5, 1.0, .529, .702, .471),
       ball(2.0, 1.5, 1.8, .278, .573, .710),
       ball(-2.0, 0.8, 1.2, .302, .419, .412),
       ball(-1.75, 0.14, 0.8, .529, .702, .471),
       ball(1.2, 1.9, 1.0, .278, .573, .710),
       ball(-1.2, 0.4, 2.0, .302, .419, .412),
       ball(1.34, 3.0, 1.3, .529, .702, .471),
       ball(-0.3, 2.3, 2.2, .278, .573, .710),
       ball(0.4, 1.2, 2.5, .302, .419, .412),
       ball(-0.6, 0.6, 0.6, .529, .702, .471),
       ball(-1.4, 3.3, 1.2, .278, .573, .710),
       plane(0., -.5, 0., 0., 1., 0.)]

ratio = float(WIDTH)/HEIGHT
screen = (-1., -1./ratio + .25, 1., 1./ratio + .25)

print('This will take awhile...')
for j, y in enumerate(np.linspace(screen[1], screen[3], HEIGHT)):
    for i, x in enumerate(np.linspace(screen[0], screen[2], WIDTH)):
        curr_color[:]=0 #reset current color
        eye_direction[:2]=(x,y)
        norm_vec=normalize(eye_direction-eye_pos)
        reflect=0
        eye=eye_pos
        eye_dir=norm_vec
        reflection = 1.0
        while reflect < MAX_REFLECT:
            trace = ray_trace(eye, eye_dir)
            if not trace:
                break
            object, intersect_point, surf_normal, col_ray= trace
            #make a new ray
            eye=intersect_point+surf_normal*0.0001
            eye_dir=normalize(eye_dir - 2 * np.dot(eye_dir, surf_normal) * surf_normal)
            reflect+=1
            curr_color+=reflection*col_ray
            reflection*=object.reflection
        img[HEIGHT-j-1, i, :]=np.clip(curr_color, 0, 1)

print('Done! raytrace.png generated.')
plotting.imsave('raytrace.png', img)