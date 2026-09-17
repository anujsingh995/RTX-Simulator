import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import moderngl
import numpy as np

# ==========================================
# ULTRA RTX STYLE GPU RENDERER
# ==========================================

WIDTH = 2560
HEIGHT = 1440

pygame.init()

pygame.display.set_mode(
    (WIDTH, HEIGHT),
    DOUBLEBUF | OPENGL
)

pygame.display.set_caption("Ultra RTX Renderer")

clock = pygame.time.Clock()

# ==========================================
# MODERNGL CONTEXT
# ==========================================

ctx = moderngl.create_context()

print("GPU:", ctx.info["GL_RENDERER"])
print("OpenGL:", ctx.info["GL_VERSION"])

# ==========================================
# FULLSCREEN QUAD
# ==========================================

quad = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0,  1.0,
], dtype='f4')

vbo = ctx.buffer(quad.tobytes())

# ==========================================
# VERTEX SHADER
# ==========================================

VERTEX_SHADER = '''
#version 330

in vec2 in_vert;
out vec2 uv;

void main() {

    uv = in_vert * 0.5 + 0.5;

    gl_Position = vec4(in_vert, 0.0, 1.0);
}
'''

# ==========================================
# FRAGMENT SHADER
# ==========================================

FRAGMENT_SHADER = '''
#version 330

out vec4 fragColor;

in vec2 uv;

uniform float time;
uniform vec2 mouse;
uniform vec2 resolution;

#define MAX_STEPS 500
#define MAX_DIST 150.0
#define SURF_DIST 0.0005

// ==========================================
// SDF PRIMITIVES
// ==========================================

float sphereSDF(vec3 p, vec3 center, float radius) {
    return length(p - center) - radius;
}

float boxSDF(vec3 p, vec3 b) {

    vec3 q = abs(p) - b;

    return length(max(q,0.0)) +
           min(max(q.x,max(q.y,q.z)),0.0);
}

float planeSDF(vec3 p) {
    return p.y + 1.5;
}

// ==========================================
// SCENE
// ==========================================

float sceneSDF(vec3 p) {

    vec3 movingSphere = vec3(
        sin(time * 0.7) * 2.5,
        sin(time * 1.2) * 0.5,
        cos(time * 0.7) * 2.5
    );

    float s1 =
        sphereSDF(
            p,
            movingSphere,
            1.0
        );

    float s2 =
        sphereSDF(
            p,
            vec3(-3.5, 0.0, 4.0),
            1.2
        );

    float s3 =
        sphereSDF(
            p,
            vec3(3.5, 0.0, 4.0),
            1.2
        );

    float box1 =
        boxSDF(
            p - vec3(0.0, 0.5, 7.0),
            vec3(1.2)
        );

    float floorPlane = planeSDF(p);

    return min(
        min(min(s1,s2),min(s3,box1)),
        floorPlane
    );
}

// ==========================================
// RAY MARCHING
// ==========================================

float rayMarch(vec3 ro, vec3 rd) {

    float dO = 0.0;

    for(int i = 0; i < MAX_STEPS; i++) {

        vec3 p = ro + rd * dO;

        float dS = sceneSDF(p);

        dO += dS;

        if(dO > MAX_DIST || abs(dS) < SURF_DIST)
            break;
    }

    return dO;
}

// ==========================================
// NORMALS
// ==========================================

vec3 getNormal(vec3 p) {

    vec2 e = vec2(0.001, 0.0);

    float d = sceneSDF(p);

    vec3 n = d - vec3(
        sceneSDF(p - e.xyy),
        sceneSDF(p - e.yxy),
        sceneSDF(p - e.yyx)
    );

    return normalize(n);
}

// ==========================================
// SOFT SHADOWS
// ==========================================

float softShadow(vec3 ro, vec3 rd) {

    float res = 1.0;

    float t = 0.02;

    for(int i = 0; i < 64; i++) {

        float h = sceneSDF(ro + rd * t);

        res = min(res, 8.0 * h / t);

        t += clamp(h, 0.02, 0.5);

        if(h < 0.001 || t > 50.0)
            break;
    }

    return clamp(res, 0.0, 1.0);
}

// ==========================================
// AMBIENT OCCLUSION
// ==========================================

float ambientOcclusion(vec3 p, vec3 n) {

    float occ = 0.0;

    float sca = 1.0;

    for(int i = 0; i < 5; i++) {

        float h = 0.01 + 0.12 * float(i) / 4.0;

        float d = sceneSDF(p + n * h);

        occ += (h - d) * sca;

        sca *= 0.95;
    }

    return clamp(1.0 - occ, 0.0, 1.0);
}

// ==========================================
// LIGHTING
// ==========================================

vec3 lighting(vec3 p, vec3 rd, vec3 baseColor) {

    vec3 n = getNormal(p);

    vec3 lightPos = vec3(
        sin(time) * 8.0,
        8.0,
        cos(time) * 8.0
    );

    vec3 l = normalize(lightPos - p);

    float diff = max(dot(n, l), 0.0);

    float shadow =
        softShadow(
            p + n * 0.01,
            l
        );

    float ao =
        ambientOcclusion(
            p,
            n
        );

    vec3 refl = reflect(-l, n);

    float spec =
        pow(
            max(dot(refl, -rd), 0.0),
            128.0
        );

    vec3 color =
        baseColor * diff * shadow;

    color += spec * 1.2;

    color *= ao;

    return color;
}

// ==========================================
// REFLECTION TRACE
// ==========================================

vec3 trace(vec3 ro, vec3 rd) {

    vec3 finalColor = vec3(0.0);

    float reflection = 1.0;

    for(int bounce = 0; bounce < 5; bounce++) {

        float d = rayMarch(ro, rd);

        if(d > MAX_DIST)
            break;

        vec3 p = ro + rd * d;

        vec3 n = getNormal(p);

        vec3 baseColor;

        // FLOOR

        if(p.y < -1.4) {

            float checker =
                mod(
                    floor(p.x) +
                    floor(p.z),
                    2.0
                );

            baseColor = mix(
                vec3(0.03),
                vec3(0.9),
                checker
            );
        }

        // OBJECTS

        else {

            baseColor = vec3(
                0.2 + float(bounce) * 0.1,
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

        reflection *= 0.55;

        rd = reflect(rd, n);

        ro = p + n * 0.02;
    }

    return finalColor;
}

// ==========================================
// MAIN
// ==========================================

void main() {

    vec2 p = uv * 2.0 - 1.0;

    p.x *= resolution.x / resolution.y;

    // CAMERA

    vec3 ro = vec3(
        mouse.x * 6.0,
        2.5 + mouse.y * 3.0,
        -12.0
    );

    vec3 rd =
        normalize(
            vec3(p.x, p.y, 1.7)
        );

    // CAMERA ROTATION

    float angle = time * 0.2;

    mat2 rot = mat2(
        cos(angle), -sin(angle),
        sin(angle),  cos(angle)
    );

    rd.xz *= rot;

    // TRACE

    vec3 color =
        trace(ro, rd);

    // FOG

    float fog =
        exp(-0.015 * length(rd));

    color =
        mix(
            vec3(0.01,0.015,0.03),
            color,
            fog
        );

    // NEON GLOW

    color += vec3(
        0.02,
        0.01,
        0.05
    );

    // GAMMA

    color =
        pow(
            color,
            vec3(0.4545)
        );

    fragColor = vec4(color, 1.0);
}
'''

# ==========================================
# COMPILE SHADERS
# ==========================================

try:

    prog = ctx.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader=FRAGMENT_SHADER
    )

except Exception as e:

    print(e)
    quit()

# ==========================================
# VAO
# ==========================================

vao = ctx.simple_vertex_array(
    prog,
    vbo,
    'in_vert'
)

prog['resolution'].value = (WIDTH, HEIGHT)

# ==========================================
# MAIN LOOP
# ==========================================

running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

    mx, my = pygame.mouse.get_pos()

    mx = (mx / WIDTH) - 0.5
    my = -((my / HEIGHT) - 0.5)

    prog['mouse'].value = (mx, my)

    prog['time'].value = (
        pygame.time.get_ticks() / 1000.0
    )

    ctx.clear(0.0, 0.0, 0.0)

    vao.render(moderngl.TRIANGLE_STRIP)

    pygame.display.flip()

    clock.tick(165)

pygame.quit()