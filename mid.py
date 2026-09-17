import pygame
from pygame.locals import *
import moderngl
import numpy as np
import multiprocessing
import time
import os

WIDTH = 1280
HEIGHT = 720

RUN_TIME = 40

# ==========================================================
# CPU WORKER
# ==========================================================

def cpu_worker(shared_flag):

    particles = np.random.rand(
        30000,
        6
    ).astype(np.float32)

    while shared_flag.value:

        particles[:,0] += particles[:,3] * 0.01
        particles[:,1] += particles[:,4] * 0.01
        particles[:,2] += particles[:,5] * 0.01

        particles[:,4] -= 0.00005

        mask = particles[:,1] < -50

        particles[mask,1] = 50

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    pygame.init()

    pygame.display.set_mode(
        (WIDTH, HEIGHT),
        DOUBLEBUF | OPENGL
    )

    clock = pygame.time.Clock()

    ctx = moderngl.create_context()

    print("GPU:", ctx.info["GL_RENDERER"])
    print("OpenGL:", ctx.info["GL_VERSION"])

    # ======================================================
    # CPU LOAD
    # ======================================================

    cpu_running = multiprocessing.Value('b', True)

    CPU_COUNT = 2

    cpu_processes = []

    for i in range(CPU_COUNT):

        p = multiprocessing.Process(
            target=cpu_worker,
            args=(cpu_running,)
        )

        p.start()

        cpu_processes.append(p)

    print("CPU Processes:", CPU_COUNT)

    # ======================================================
    # FULLSCREEN QUAD
    # ======================================================

    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype='f4')

    vbo = ctx.buffer(vertices.tobytes())

    # ======================================================
    # SHADERS
    # ======================================================

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

    FRAGMENT_SHADER = '''
    #version 330

    out vec4 fragColor;

    in vec2 uv;

    uniform float time;
    uniform vec2 resolution;

    #define MAX_STEPS 500
    #define MAX_DIST 100.0
    #define SURF_DIST 0.001

    mat2 rot(float a) {

        float s = sin(a);
        float c = cos(a);

        return mat2(c,-s,s,c);
    }

    float sphereSDF(
        vec3 p,
        vec3 c,
        float r
    ) {

        return length(p-c)-r;
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

    float sceneSDF(vec3 p) {

        vec3 pp = p;

        pp.xz *= rot(time * 0.2);

        float sphere1 =
            sphereSDF(
                pp,
                vec3(
                    sin(time)*4.0,
                    0.0,
                    cos(time)*4.0
                ),
                1.5
            );

        float torus1 =
            torusSDF(
                pp - vec3(0.0,0.0,8.0),
                vec2(2.0,0.5)
            );

        float floorPlane =
            p.y + 2.0;

        return min(
            min(sphere1,torus1),
            floorPlane
        );
    }

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

    void main() {

        vec2 p =
            uv * 2.0 - 1.0;

        p.x *=
            resolution.x
            / resolution.y;

        vec3 ro = vec3(
            sin(time*0.2)*8.0,
            3.0,
            -18.0
        );

        vec3 rd =
            normalize(
                vec3(
                    p.x,
                    p.y,
                    1.8
                )
            );

        float d =
            rayMarch(ro,rd);

        vec3 color = vec3(0.0);

        if(d < MAX_DIST) {

            vec3 pos =
                ro + rd*d;

            vec3 n =
                getNormal(pos);

            vec3 light =
                normalize(
                    vec3(5.0,8.0,-5.0)
                );

            float diff =
                max(dot(n,light),0.0);

            color =
                vec3(
                    0.2,
                    0.5,
                    1.0
                ) * diff;
        }

        color += vec3(
            0.02,
            0.01,
            0.04
        );

        color =
            pow(
                color,
                vec3(0.4545)
            );

        fragColor =
            vec4(color,1.0);
    }
    '''

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

        clock.tick()

        fps = clock.get_fps()

        pygame.display.set_caption(
            f"STABLE RTX ENGINE | FPS: {fps:.2f}"
        )

        pygame.display.flip()

    cpu_running.value = False

    for p in cpu_processes:

        p.terminate()

    pygame.quit()