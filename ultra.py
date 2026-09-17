import pygame
from pygame.locals import *
import moderngl
import numpy as np
import time
import math
import multiprocessing
import os
import random

# ==========================================================
# HYBRID RTX CINEMATIC ENGINE V2
# GPU + CPU Procedural Benchmark
# Stable Cinematic Renderer
# Native 1080P
# ESC TO EXIT
# ==========================================================

WIDTH = 1920
HEIGHT = 1080

RUN_TIME = 40

pygame.init()

pygame.display.set_mode(
    (WIDTH, HEIGHT),
    DOUBLEBUF | OPENGL
)

clock = pygame.time.Clock()

ctx = moderngl.create_context()

print("GPU:", ctx.info["GL_RENDERER"])
print("OpenGL:", ctx.info["GL_VERSION"])

# ==========================================================
# CONTROLLED CPU LOAD
# ==========================================================

cpu_running = multiprocessing.Value('b', True)

def cpu_worker(shared_flag):

    particles = np.random.rand(120000, 6).astype(np.float32)

    while shared_flag.value:

        # physics simulation

        particles[:,0] += particles[:,3] * 0.01
        particles[:,1] += particles[:,4] * 0.01
        particles[:,2] += particles[:,5] * 0.01

        # gravity

        particles[:,4] -= 0.00005

        # turbulence

        particles[:,3] += np.sin(
            particles[:,1]
        ) * 0.0001

        particles[:,5] += np.cos(
            particles[:,0]
        ) * 0.0001

        # bounds reset

        mask = particles[:,1] < -50

        particles[mask,1] = 50

# USE HALF CORES FOR STABILITY

CPU_COUNT = max(
    2,
    os.cpu_count() // 2
)

cpu_processes = []

for i in range(CPU_COUNT):

    p = multiprocessing.Process(
        target=cpu_worker,
        args=(cpu_running,)
    )

    p.start()

    cpu_processes.append(p)

print("CPU Processes:", CPU_COUNT)

# ==========================================================
# FULLSCREEN QUAD
# ==========================================================

vertices = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0,  1.0,
], dtype='f4')

vbo = ctx.buffer(vertices.tobytes())

# ==========================================================
# VERTEX SHADER
# ==========================================================

VERTEX_SHADER = '''
#version 330

in vec2 in_vert;

out vec2 uv;

void main() {

    uv = in_vert * 0.5 + 0.5;

    gl_Position = vec4(
        in_vert,
        0.0,
        1.0
    );
}
'''

# ==========================================================
# FRAGMENT SHADER
# ==========================================================

FRAGMENT_SHADER = '''
#version 330

out vec4 fragColor;

in vec2 uv;

uniform float time;
uniform vec2 resolution;

#define MAX_STEPS 950
#define MAX_DIST 150.0
#define SURF_DIST 0.0008

// ==========================================================
// ROTATION
// ==========================================================

mat2 rot(float a) {

    float s = sin(a);
    float c = cos(a);

    return mat2(c,-s,s,c);
}

// ==========================================================
// SDF OBJECTS
// ==========================================================

float sphereSDF(
    vec3 p,
    vec3 c,
    float r
) {

    return length(p-c)-r;
}

float boxSDF(
    vec3 p,
    vec3 b
) {

    vec3 q = abs(p)-b;

    return length(max(q,0.0))
        + min(
            max(q.x,max(q.y,q.z)),
            0.0
        );
}

float torusSDF(
    vec3 p,
    vec2 t
) {

    vec2 q = vec2(
        length(p.xz)-t.x,
        p.y
    );

    return length(q)-t.y;
}

// ==========================================================
// FRACTAL
// ==========================================================

float fractal(vec3 p) {

    vec3 z = p;

    float dr = 1.0;
    float r = 0.0;

    for(int i=0;i<7;i++) {

        r = length(z);

        if(r > 2.0)
            break;

        float theta =
            acos(z.z/r);

        float phi =
            atan(z.y,z.x);

        dr =
            pow(r,7.0)
            * 8.0
            * dr
            + 1.0;

        float zr =
            pow(r,8.0);

        theta *= 8.0;
        phi *= 8.0;

        z = zr * vec3(
            sin(theta)*cos(phi),
            sin(phi)*sin(theta),
            cos(theta)
        );

        z += p;
    }

    return 0.5
        * log(r)
        * r
        / dr;
}

// ==========================================================
// SCENE
// ==========================================================

float sceneSDF(vec3 p) {

    vec3 pp = p;

    pp.xz *= rot(time * 0.15);

    float floorPlane =
        p.y + 2.0;

    float sphere1 =
        sphereSDF(
            pp,
            vec3(
                sin(time)*5.0,
                sin(time*1.5),
                cos(time)*5.0
            ),
            1.8
        );

    float sphere2 =
        sphereSDF(
            pp,
            vec3(-6.0,1.0,12.0),
            2.0
        );

    float torus1 =
        torusSDF(
            pp - vec3(6.0,0.0,10.0),
            vec2(2.5,0.5)
        );

    float box1 =
        boxSDF(
            pp - vec3(0.0,4.0,18.0),
            vec3(2.5)
        );

    float fract =
        fractal(
            pp - vec3(0.0,0.0,25.0)
        );

    float objects =
        min(
            min(sphere1,sphere2),
            min(
                torus1,
                min(box1,fract)
            )
        );

    return min(
        objects,
        floorPlane
    );
}

// ==========================================================
// RAY MARCHING
// ==========================================================

float rayMarch(
    vec3 ro,
    vec3 rd
) {

    float dO = 0.0;

    for(int i=0;i<MAX_STEPS;i++) {

        vec3 p =
            ro + rd*dO;

        float dS =
            sceneSDF(p);

        dO += dS;

        if(
            dO > MAX_DIST
            || abs(dS) < SURF_DIST
        )
            break;
    }

    return dO;
}

// ==========================================================
// NORMALS
// ==========================================================

vec3 getNormal(vec3 p) {

    vec2 e = vec2(0.001,0);

    float d = sceneSDF(p);

    vec3 n = d - vec3(
        sceneSDF(p-e.xyy),
        sceneSDF(p-e.yxy),
        sceneSDF(p-e.yyx)
    );

    return normalize(n);
}

// ==========================================================
// SHADOWS
// ==========================================================

float softShadow(
    vec3 ro,
    vec3 rd
) {

    float res = 1.0;

    float t = 0.02;

    for(int i=0;i<48;i++) {

        float h =
            sceneSDF(ro+rd*t);

        res = min(
            res,
            10.0*h/t
        );

        t += clamp(
            h,
            0.02,
            0.5
        );

        if(
            h < 0.001
            || t > 80.0
        )
            break;
    }

    return clamp(
        res,
        0.0,
        1.0
    );
}

// ==========================================================
// LIGHTING
// ==========================================================

vec3 lighting(
    vec3 p,
    vec3 rd,
    vec3 baseColor
) {

    vec3 n =
        getNormal(p);

    vec3 lightPos =
        vec3(
            sin(time)*14.0,
            12.0,
            cos(time)*14.0
        );

    vec3 l =
        normalize(lightPos-p);

    float diff =
        max(dot(n,l),0.0);

    float shadow =
        softShadow(
            p+n*0.02,
            l
        );

    vec3 refl =
        reflect(-l,n);

    float spec =
        pow(
            max(dot(refl,-rd),0.0),
            128.0
        );

    vec3 color =
        baseColor
        * diff
        * shadow;

    color += spec;

    return color;
}

// ==========================================================
// TRACE
// ==========================================================

vec3 trace(
    vec3 ro,
    vec3 rd
) {

    vec3 finalColor =
        vec3(0.0);

    float reflection = 1.0;

    for(int bounce=0;bounce<6;bounce++) {

        float d =
            rayMarch(ro,rd);

        if(d > MAX_DIST)
            break;

        vec3 p =
            ro + rd*d;

        vec3 n =
            getNormal(p);

        vec3 baseColor;

        if(p.y < -1.9) {

            float checker =
                mod(
                    floor(p.x)
                    + floor(p.z),
                    2.0
                );

            baseColor =
                mix(
                    vec3(0.03),
                    vec3(0.9),
                    checker
                );
        }

        else {

            baseColor =
                vec3(
                    0.2
                    + float(bounce)*0.05,
                    0.5,
                    1.0
                );
        }

        vec3 light =
            lighting(
                p,
                rd,
                baseColor
            );

        finalColor +=
            light * reflection;

        reflection *= 0.6;

        rd = reflect(rd,n);

        ro = p + n*0.03;
    }

    return finalColor;
}

// ==========================================================
// MAIN
// ==========================================================

void main() {

    vec2 p =
        uv * 2.0 - 1.0;

    p.x *=
        resolution.x
        / resolution.y;

    vec3 ro = vec3(
        sin(time*0.2)*8.0,
        4.0 + sin(time*0.35),
        -20.0
    );

    // CAMERA SHAKE

    ro.x +=
        sin(time*13.0)*0.04;

    ro.y +=
        cos(time*15.0)*0.04;

    float angle =
        time * 0.2;

    mat2 r = rot(angle);

    vec3 color =
        vec3(0.0);

    // SUPERSAMPLING

    for(int x=-1;x<=1;x++) {

        for(int y=-1;y<=1;y++) {

            vec2 offset =
                vec2(
                    float(x),
                    float(y)
                ) / resolution;

            vec3 rd =
                normalize(
                    vec3(
                        p.x + offset.x,
                        p.y + offset.y,
                        1.8
                    )
                );

            rd.xz *= r;

            color += trace(ro,rd);
        }
    }

    color /= 9.0;

    // FOG

    float fogAccum = 0.0;

    for(int i=0;i<28;i++) {

        float fi =
            float(i)/28.0;

        vec3 fogPos =
            ro + vec3(
                p,
                fi*80.0
            );

        fogAccum +=
            exp(-fogPos.y*0.14)
            * 0.02;
    }

    color += vec3(
        0.18,
        0.08,
        0.35
    ) * fogAccum;

    // BLOOM

    color +=
        pow(
            color,
            vec3(2.0)
        ) * 0.2;

    // GAMMA

    color =
        pow(
            color,
            vec3(0.4545)
        );

    fragColor =
        vec4(color,1.0);
}
'''

# ==========================================================
# PROGRAM
# ==========================================================

prog = ctx.program(
    vertex_shader=VERTEX_SHADER,
    fragment_shader=FRAGMENT_SHADER
)

vao = ctx.simple_vertex_array(
    prog,
    vbo,
    'in_vert'
)

prog['resolution'].value = (
    WIDTH,
    HEIGHT
)

# ==========================================================
# MAIN LOOP
# ==========================================================

start_time = time.time()

running = True

while running:

    current_time = (
        time.time()
        - start_time
    )

    if current_time >= RUN_TIME:
        running = False

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                running = False

    prog['time'].value = current_time

    ctx.clear(0.0,0.0,0.0)

    vao.render(moderngl.TRIANGLE_STRIP)

    dt = clock.tick() / 1000.0

    fps = (
        1.0 / dt
        if dt > 0
        else 0
    )

    title = (
        f"HYBRID RTX ENGINE V2 | "
        f"FPS: {fps:.2f} | "
        f"CPU PROC: {CPU_COUNT}"
    )

    pygame.display.set_caption(title)

    print(title)

    pygame.display.flip()

# ==========================================================
# CLEANUP
# ==========================================================

cpu_running.value = False

for p in cpu_processes:

    p.terminate()

pygame.quit()