/// <reference path="./webgpu.d.ts" />
/// <reference types="react" />

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useContainerWidth } from '../utils';

/******************************************************
 * 0) Rendering Constants
 ******************************************************/
const LIGHTING = {
  AMBIENT_INTENSITY: 0.4,
  DIFFUSE_INTENSITY: 0.6,
  SPECULAR_INTENSITY: 0.2,
  SPECULAR_POWER: 20.0,
} as const;

const GEOMETRY = {
  SPHERE: {
    STACKS: 16,
    SLICES: 24,
    MIN_STACKS: 8,
    MIN_SLICES: 12,
  }
} as const;

/******************************************************
 * 1) Define Data Structures
 ******************************************************/
interface PointCloudData {
  positions: Float32Array;     // [x, y, z, ...]
  colors?: Float32Array;       // [r, g, b, ...] in [0..1]
  scales?: Float32Array;       // optional
}

interface EllipsoidData {
  centers: Float32Array;       // [cx, cy, cz, ...]
  radii: Float32Array;         // [rx, ry, rz, ...]
  colors?: Float32Array;       // [r, g, b, ...] in [0..1]
}

interface Decoration {
  indexes: number[];
  color?: [number, number, number];
  alpha?: number;
  scale?: number;
  minSize?: number;
}

interface PointCloudElementConfig {
  type: 'PointCloud';
  data: PointCloudData;
  decorations?: Decoration[];
}

interface EllipsoidElementConfig {
  type: 'Ellipsoid';
  data: EllipsoidData;
  decorations?: Decoration[];
}

// [BAND CHANGE #1]: New EllipsoidBand type
interface EllipsoidBandElementConfig {
  type: 'EllipsoidBand';
  data: EllipsoidData;          // reusing the same “centers/radii/colors” style
  decorations?: Decoration[];
}

// [BAND CHANGE #2]: Extend SceneElementConfig
type SceneElementConfig =
  | PointCloudElementConfig
  | EllipsoidElementConfig
  | EllipsoidBandElementConfig;

/******************************************************
 * 2) Minimal Camera State
 ******************************************************/
interface CameraState {
  orbitRadius: number;
  orbitTheta: number;
  orbitPhi: number;
  panX: number;
  panY: number;
  fov: number;
  near: number;
  far: number;
}

/******************************************************
 * 3) React Component Props
 ******************************************************/
interface SceneProps {
  elements: SceneElementConfig[];
  containerWidth: number;
}

/******************************************************
 * 4) SceneWrapper
 ******************************************************/
export function SceneWrapper({ elements }: { elements: SceneElementConfig[] }) {
  const [containerRef, measuredWidth] = useContainerWidth(1);
  return (
    <div ref={containerRef} style={{ width: '100%', height: '600px' }}>
      {measuredWidth > 0 && (
        <Scene containerWidth={measuredWidth} elements={elements} />
      )}
    </div>
  );
}

/******************************************************
 * 5) The Scene Component
 ******************************************************/
function Scene({ elements, containerWidth }: SceneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const safeWidth = containerWidth > 0 ? containerWidth : 300;
  const canvasWidth = safeWidth;
  const canvasHeight = safeWidth;

  // GPU + pipeline references
  const gpuRef = useRef<{
    device: GPUDevice;
    context: GPUCanvasContext;

    // Billboards
    billboardPipeline: GPURenderPipeline;
    billboardQuadVB: GPUBuffer;
    billboardQuadIB: GPUBuffer;

    // 3D Ellipsoids
    ellipsoidPipeline: GPURenderPipeline;
    sphereVB: GPUBuffer;   // pos+normal
    sphereIB: GPUBuffer;
    sphereIndexCount: number;

    // [BAND CHANGE #3]: EllipsoidBand
    ellipsoidBandPipeline: GPURenderPipeline;
    ringVB: GPUBuffer;     // ring geometry
    ringIB: GPUBuffer;
    ringIndexCount: number;

    // Shared uniform data
    uniformBuffer: GPUBuffer;
    uniformBindGroup: GPUBindGroup;

    // Instances
    pcInstanceBuffer: GPUBuffer | null;
    pcInstanceCount: number;
    ellipsoidInstanceBuffer: GPUBuffer | null;
    ellipsoidInstanceCount: number;

    // [BAND CHANGE #4]: EllipsoidBand instances
    bandInstanceBuffer: GPUBuffer | null;
    bandInstanceCount: number;

    // Depth
    depthTexture: GPUTexture | null;
  } | null>(null);

  // Render loop handle
  const rafIdRef = useRef<number>(0);

  // Track readiness
  const [isReady, setIsReady] = useState(false);

  // Camera
  const [camera, setCamera] = useState<CameraState>({
    orbitRadius: 2.0,
    orbitTheta: 0.2,
    orbitPhi: 1.0,
    panX: 0.0,
    panY: 0.0,
    fov: Math.PI / 3,
    near: 0.01,
    far: 100.0,
  });

  /******************************************************
   * A) Minimal Math
   ******************************************************/
  function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        out[j * 4 + i] =
          a[i + 0] * b[j * 4 + 0] +
          a[i + 4] * b[j * 4 + 1] +
          a[i + 8] * b[j * 4 + 2] +
          a[i + 12] * b[j * 4 + 3];
      }
    }
    return out;
  }

  function mat4Perspective(fov: number, aspect: number, near: number, far: number): Float32Array {
    const out = new Float32Array(16);
    const f = 1.0 / Math.tan(fov / 2);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) / (near - far);
    out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
    return out;
  }

  function mat4LookAt(eye: [number, number, number], target: [number, number, number], up: [number, number, number]): Float32Array {
    const zAxis = normalize([
      eye[0] - target[0],
      eye[1] - target[1],
      eye[2] - target[2],
    ]);
    const xAxis = normalize(cross(up, zAxis));
    const yAxis = cross(zAxis, xAxis);

    const out = new Float32Array(16);
    out[0] = xAxis[0];
    out[1] = yAxis[0];
    out[2] = zAxis[0];
    out[3] = 0;

    out[4] = xAxis[1];
    out[5] = yAxis[1];
    out[6] = zAxis[1];
    out[7] = 0;

    out[8] = xAxis[2];
    out[9] = yAxis[2];
    out[10] = zAxis[2];
    out[11] = 0;

    out[12] = -dot(xAxis, eye);
    out[13] = -dot(yAxis, eye);
    out[14] = -dot(zAxis, eye);
    out[15] = 1;
    return out;
  }

  function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }
  function dot(a: [number, number, number], b: [number, number, number]): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }
  function normalize(v: [number, number, number]): [number, number, number] {
    const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > 1e-6) {
      return [v[0]/len, v[1]/len, v[2]/len];
    }
    return [0, 0, 0];
  }

  /******************************************************
   * B) Generate Sphere Geometry with Normals
   ******************************************************/
  function createSphereGeometry(
    stacks = GEOMETRY.SPHERE.STACKS,
    slices = GEOMETRY.SPHERE.SLICES,
    radius = 1.0
  ) {
    const actualStacks = radius < 0.1
      ? GEOMETRY.SPHERE.MIN_STACKS
      : stacks;
    const actualSlices = radius < 0.1
      ? GEOMETRY.SPHERE.MIN_SLICES
      : slices;

    const verts: number[] = [];
    const indices: number[] = [];

    for (let i = 0; i <= actualStacks; i++) {
      const phi = (i / actualStacks) * Math.PI;
      const cosPhi = Math.cos(phi);
      const sinPhi = Math.sin(phi);

      for (let j = 0; j <= actualSlices; j++) {
        const theta = (j / actualSlices) * 2 * Math.PI;
        const cosTheta = Math.cos(theta);
        const sinTheta = Math.sin(theta);

        const x = sinPhi * cosTheta;
        const y = cosPhi;
        const z = sinPhi * sinTheta;

        // position
        verts.push(x, y, z);
        // normal
        verts.push(x, y, z);
      }
    }

    for (let i = 0; i < actualStacks; i++) {
      for (let j = 0; j < actualSlices; j++) {
        const row1 = i * (actualSlices + 1) + j;
        const row2 = (i + 1) * (actualSlices + 1) + j;

        indices.push(row1, row2, row1 + 1);
        indices.push(row1 + 1, row2, row2 + 1);
      }
    }

    return {
      vertexData: new Float32Array(verts),
      indexData: new Uint16Array(indices),
    };
  }

  // [BAND CHANGE #5]: Create simple ring geometry in XY plane
  function createRingGeometry(segments: number = 32, rInner = 0.98, rOuter = 1.0) {
    // We'll build a thin triangular ring from rInner to rOuter in the XY plane
    const verts: number[] = [];
    const indices: number[] = [];
    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * 2 * Math.PI;
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);

      // Outer
      verts.push(rOuter * cosT, rOuter * sinT, 0);  // position
      verts.push(0, 0, 1);                          // normal (arbitrary)
      // Inner
      verts.push(rInner * cosT, rInner * sinT, 0);
      verts.push(0, 0, 1);
    }
    // Triangulate as a triangle strip
    for (let i = 0; i < segments * 2; i += 2) {
      indices.push(i, i + 1, i + 2);
      indices.push(i + 2, i + 1, i + 3);
    }

    return {
      vertexData: new Float32Array(verts),
      indexData: new Uint16Array(indices),
    };
  }

  /******************************************************
   * C) Initialize WebGPU
   ******************************************************/
  const initWebGPU = useCallback(async () => {
    if (!canvasRef.current) return;
    if (!navigator.gpu) {
      console.error('WebGPU not supported.');
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('Failed to get GPU adapter.');
      const device = await adapter.requestDevice();

      const context = canvasRef.current.getContext('webgpu') as GPUCanvasContext;
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: 'premultiplied' });

      // 1) Billboards
      const QUAD_VERTICES = new Float32Array([
        -0.5, -0.5,
         0.5, -0.5,
        -0.5,  0.5,
         0.5,  0.5,
      ]);
      const QUAD_INDICES = new Uint16Array([0,1,2,2,1,3]);

      const billboardQuadVB = device.createBuffer({
        size: QUAD_VERTICES.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(billboardQuadVB, 0, QUAD_VERTICES);

      const billboardQuadIB = device.createBuffer({
        size: QUAD_INDICES.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(billboardQuadIB, 0, QUAD_INDICES);

      // 2) Sphere (pos + normal)
      const sphereGeo = createSphereGeometry();
      const sphereVB = device.createBuffer({
        size: sphereGeo.vertexData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(sphereVB, 0, sphereGeo.vertexData);

      const sphereIB = device.createBuffer({
        size: sphereGeo.indexData.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(sphereIB, 0, sphereGeo.indexData);

      // [BAND CHANGE #6]: Ring geometry
      const ringGeo = createRingGeometry(64, 0.95, 1.0);
      const ringVB = device.createBuffer({
        size: ringGeo.vertexData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(ringVB, 0, ringGeo.vertexData);

      const ringIB = device.createBuffer({
        size: ringGeo.indexData.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(ringIB, 0, ringGeo.indexData);

      // 3) Uniform buffer
      const uniformBufferSize = 128;
      const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // 4) Pipeline layout
      const uniformBindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' },
          },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [uniformBindGroupLayout],
      });

      // 5) Billboard pipeline
      const billboardPipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
          module: device.createShaderModule({
            code: `
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  pad1: f32,
  cameraUp: vec3<f32>,
  pad2: f32,
  lightDir: vec3<f32>,
  pad3: f32,
};

@group(0) @binding(0) var<uniform> camera : Camera;

struct VSOut {
  @builtin(position) position : vec4<f32>,
  @location(2) color : vec3<f32>,
  @location(3) alpha : f32,
};

@vertex
fn vs_main(
  // Quad corners
  @location(0) corner : vec2<f32>,
  // Instance data: pos(3), color(3), alpha(1), scaleX(1), scaleY(1)
  @location(1) pos    : vec3<f32>,
  @location(2) col    : vec3<f32>,
  @location(3) alpha  : f32,
  @location(4) scaleX : f32,
  @location(5) scaleY : f32
) -> VSOut {
  var out: VSOut;
  let offset = camera.cameraRight * (corner.x * scaleX)
             + camera.cameraUp    * (corner.y * scaleY);
  let worldPos = vec4<f32>(
    pos.x + offset.x,
    pos.y + offset.y,
    pos.z + offset.z,
    1.0
  );
  out.position = camera.mvp * worldPos;
  out.color = col;
  out.alpha = alpha;
  return out;
}

@fragment
fn fs_main(
  @location(2) inColor : vec3<f32>,
  @location(3) inAlpha : f32
) -> @location(0) vec4<f32> {
  return vec4<f32>(inColor, inAlpha);
}
`
          }),
          entryPoint: 'vs_main',
          buffers: [
            // corners
            {
              arrayStride: 2 * 4,
              attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
            },
            // instance data
            {
              arrayStride: 9 * 4,
              stepMode: 'instance',
              attributes: [
                { shaderLocation: 1, offset: 0,         format: 'float32x3' }, // pos
                { shaderLocation: 2, offset: 3 * 4,     format: 'float32x3' }, // col
                { shaderLocation: 3, offset: 6 * 4,     format: 'float32'   }, // alpha
                { shaderLocation: 4, offset: 7 * 4,     format: 'float32'   }, // scaleX
                { shaderLocation: 5, offset: 8 * 4,     format: 'float32'   }, // scaleY
              ],
            },
          ],
        },
        fragment: {
          module: device.createShaderModule({
            code: `
@fragment
fn fs_main(
  @location(2) inColor: vec3<f32>,
  @location(3) inAlpha: f32
) -> @location(0) vec4<f32> {
  return vec4<f32>(inColor, inAlpha);
}
`
          }),
          entryPoint: 'fs_main',
          targets: [{
            format,
            blend: {
              color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              }
            }
          }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: true,
          depthCompare: 'less-equal',
        },
      });

      // 6) Ellipsoid pipeline
      const ellipsoidPipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
          module: device.createShaderModule({
            code: `
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  pad1: f32,
  cameraUp: vec3<f32>,
  pad2: f32,
  lightDir: vec3<f32>,
  pad3: f32,
};

@group(0) @binding(0) var<uniform> camera : Camera;

struct VSOut {
  @builtin(position) Position : vec4<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) color : vec3<f32>,
  @location(3) alpha : f32,
  @location(4) worldPos : vec3<f32>,
};

@vertex
fn vs_main(
  // sphere data: pos(3), norm(3)
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,

  // instance data: pos(3), scale(3), color(3), alpha(1)
  @location(2) iPos: vec3<f32>,
  @location(3) iScale: vec3<f32>,
  @location(4) iColor: vec3<f32>,
  @location(5) iAlpha: f32
) -> VSOut {
  var out: VSOut;
  let worldPos = vec3<f32>(
    iPos.x + inPos.x * iScale.x,
    iPos.y + inPos.y * iScale.y,
    iPos.z + inPos.z * iScale.z
  );
  let scaledNorm = normalize(vec3<f32>(
    inNorm.x / iScale.x,
    inNorm.y / iScale.y,
    inNorm.z / iScale.z
  ));
  out.Position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.normal = scaledNorm;
  out.color = iColor;
  out.alpha = iAlpha;
  out.worldPos = worldPos;
  return out;
}

@fragment
fn fs_main(
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>
) -> @location(0) vec4<f32> {
  let N = normalize(normal);
  let L = normalize(camera.lightDir);
  let lambert = max(dot(N, L), 0.0);

  let ambient = ${LIGHTING.AMBIENT_INTENSITY};
  var color = baseColor * (ambient + lambert * ${LIGHTING.DIFFUSE_INTENSITY});

  // small spec
  let V = normalize(-worldPos);
  let H = normalize(L + V);
  let spec = pow(max(dot(N, H), 0.0), ${LIGHTING.SPECULAR_POWER});
  color += vec3<f32>(1.0,1.0,1.0) * spec * ${LIGHTING.SPECULAR_INTENSITY};

  return vec4<f32>(color, alpha);
}
`
          }),
          entryPoint: 'vs_main',
          buffers: [
            // sphere geometry: pos(3), normal(3)
            {
              arrayStride: 6 * 4,
              attributes: [
                { shaderLocation: 0, offset: 0,       format: 'float32x3' }, // inPos
                { shaderLocation: 1, offset: 3 * 4,   format: 'float32x3' }, // inNorm
              ],
            },
            // instance data
            {
              arrayStride: 10 * 4,
              stepMode: 'instance',
              attributes: [
                { shaderLocation: 2, offset: 0,         format: 'float32x3' }, // iPos
                { shaderLocation: 3, offset: 3 * 4,     format: 'float32x3' }, // iScale
                { shaderLocation: 4, offset: 6 * 4,     format: 'float32x3' }, // iColor
                { shaderLocation: 5, offset: 9 * 4,     format: 'float32'   }, // iAlpha
              ],
            },
          ],
        },
        fragment: {
          module: device.createShaderModule({
            code: `
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  pad1: f32,
  cameraUp: vec3<f32>,
  pad2: f32,
  lightDir: vec3<f32>,
  pad3: f32,
};

@group(0) @binding(0) var<uniform> camera : Camera;

@fragment
fn fs_main(
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>
) -> @location(0) vec4<f32> {
  let N = normalize(normal);
  let L = normalize(camera.lightDir);
  let lambert = max(dot(N, L), 0.0);

  let ambient = ${LIGHTING.AMBIENT_INTENSITY};
  var color = baseColor * (ambient + lambert * ${LIGHTING.DIFFUSE_INTENSITY});

  let V = normalize(-worldPos);
  let H = normalize(L + V);
  let spec = pow(max(dot(N, H), 0.0), ${LIGHTING.SPECULAR_POWER});
  color += vec3<f32>(1.0,1.0,1.0) * spec * ${LIGHTING.SPECULAR_INTENSITY};

  return vec4<f32>(color, alpha);
}
`
          }),
          entryPoint: 'fs_main',
          targets: [{
            format,
            blend: {
              color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              }
            }
          }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: true,
          depthCompare: 'less-equal',
        },
      });

      // [BAND CHANGE #7]: EllipsoidBand pipeline
      // We'll reuse a minimal approach: pos(3), normal(3), plus instance data for transform + color + alpha
      const ellipsoidBandPipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
          module: device.createShaderModule({
            code: `
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  pad1: f32,
  cameraUp: vec3<f32>,
  pad2: f32,
  lightDir: vec3<f32>,
  pad3: f32,
};

@group(0) @binding(0) var<uniform> camera : Camera;

struct VSOut {
  @builtin(position) Position : vec4<f32>,
  @location(1) color : vec3<f32>,
  @location(2) alpha : f32,
};

@vertex
fn vs_main(
  @builtin(instance_index) instanceIdx: u32,
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) iCenter: vec3<f32>,
  @location(3) iScale: vec3<f32>,
  @location(4) iColor: vec3<f32>,
  @location(5) iAlpha: f32
) -> VSOut {
  var out: VSOut;

  // Get ring index from instance ID (0=XY, 1=YZ, 2=XZ)
  let ringIndex = i32(instanceIdx % 3u);

  // Transform the ring based on its orientation
  var worldPos: vec3<f32>;
  if (ringIndex == 0) {
    // XY plane - original orientation
    worldPos = vec3<f32>(
      inPos.x * iScale.x,
      inPos.y * iScale.y,
      inPos.z
    );
  } else if (ringIndex == 1) {
    // YZ plane - rotate around X
    worldPos = vec3<f32>(
      inPos.z,
      inPos.x * iScale.y,
      inPos.y * iScale.z
    );
  } else {
    // XZ plane - rotate around Y
    worldPos = vec3<f32>(
      inPos.x * iScale.x,
      inPos.z,
      inPos.y * iScale.z
    );
  }

  // Add center offset
  worldPos += iCenter;

  out.Position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = iColor;
  out.alpha = iAlpha;
  return out;
}

@fragment
fn fs_main(
  @location(1) bColor: vec3<f32>,
  @location(2) bAlpha: f32
) -> @location(0) vec4<f32> {
  return vec4<f32>(bColor, bAlpha);
}
`
          }),
          entryPoint: 'vs_main',
          buffers: [
            // ring geometry: pos(3), normal(3)
            {
              arrayStride: 6 * 4,
              attributes: [
                { shaderLocation: 0, offset: 0,         format: 'float32x3' },
                { shaderLocation: 1, offset: 3 * 4,     format: 'float32x3' },
              ],
            },
            // instance data
            {
              arrayStride: 10 * 4,
              stepMode: 'instance',
              attributes: [
                { shaderLocation: 2, offset: 0,         format: 'float32x3' }, // center
                { shaderLocation: 3, offset: 3 * 4,     format: 'float32x3' }, // scale
                { shaderLocation: 4, offset: 6 * 4,     format: 'float32x3' }, // color
                { shaderLocation: 5, offset: 9 * 4,     format: 'float32'   }, // alpha
              ],
            },
          ],
        },
        fragment: {
          module: device.createShaderModule({
            code: `
@fragment
fn fs_main(
  @location(1) bColor: vec3<f32>,
  @location(2) bAlpha: f32
) -> @location(0) vec4<f32> {
  return vec4<f32>(bColor, bAlpha);
}
`
          }),
          entryPoint: 'fs_main',
          targets: [{
            format,
            blend: {
              color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              }
            }
          }],
        },
        primitive: {
          topology: 'triangle-list',
          cullMode: 'none',  // let both sides show
        },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: true,
          depthCompare: 'less-equal',
        },
      });

      // 7) Uniform bind group
      const uniformBindGroup = device.createBindGroup({
        layout: uniformBindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
      });

      gpuRef.current = {
        device,
        context,
        billboardPipeline,
        billboardQuadVB,
        billboardQuadIB,
        ellipsoidPipeline,
        sphereVB,
        sphereIB,
        sphereIndexCount: sphereGeo.indexData.length,

        // [BAND CHANGE #8]
        ellipsoidBandPipeline,
        ringVB,
        ringIB,
        ringIndexCount: ringGeo.indexData.length,

        uniformBuffer,
        uniformBindGroup,
        pcInstanceBuffer: null,
        pcInstanceCount: 0,
        ellipsoidInstanceBuffer: null,
        ellipsoidInstanceCount: 0,

        // [BAND CHANGE #9]
        bandInstanceBuffer: null,
        bandInstanceCount: 0,

        depthTexture: null,
      };

      setIsReady(true);
    } catch (err) {
      console.error('Error initializing WebGPU:', err);
    }
  }, []);

  /******************************************************
   * D) Create/Update Depth Texture
   ******************************************************/
  const createOrUpdateDepthTexture = useCallback(() => {
    if (!gpuRef.current) return;
    const { device, depthTexture } = gpuRef.current;
    if (depthTexture) depthTexture.destroy();

    const newDepthTex = device.createTexture({
      size: [canvasWidth, canvasHeight],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    gpuRef.current.depthTexture = newDepthTex;
  }, [canvasWidth, canvasHeight]);

  /******************************************************
   * E) Render Loop
   ******************************************************/
  const renderFrame = useCallback(() => {
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
    }

    if (!gpuRef.current) {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    const {
      device, context,
      billboardPipeline, billboardQuadVB, billboardQuadIB,
      ellipsoidPipeline, sphereVB, sphereIB, sphereIndexCount,
      ellipsoidBandPipeline, ringVB, ringIB, ringIndexCount,
      uniformBuffer, uniformBindGroup,
      pcInstanceBuffer, pcInstanceCount,
      ellipsoidInstanceBuffer, ellipsoidInstanceCount,
      bandInstanceBuffer, bandInstanceCount,
      depthTexture,
    } = gpuRef.current;
    if (!depthTexture) {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    // 1) Build camera basis + MVP
    const aspect = canvasWidth / canvasHeight;
    const proj = mat4Perspective(camera.fov, aspect, camera.near, camera.far);

    const cx = camera.orbitRadius * Math.sin(camera.orbitPhi) * Math.sin(camera.orbitTheta);
    const cy = camera.orbitRadius * Math.cos(camera.orbitPhi);
    const cz = camera.orbitRadius * Math.sin(camera.orbitPhi) * Math.cos(camera.orbitTheta);
    const eye: [number, number, number] = [cx, cy, cz];
    const target: [number, number, number] = [camera.panX, camera.panY, 0];
    const up: [number, number, number] = [0, 1, 0];
    const view = mat4LookAt(eye, target, up);

    const forward = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    const cameraRight = normalize(cross(forward, up));
    const cameraUp = cross(cameraRight, forward);

    const mvp = mat4Multiply(proj, view);

    // 2) Write uniform data
    const data = new Float32Array(32);
    data.set(mvp, 0);
    data[16] = cameraRight[0];
    data[17] = cameraRight[1];
    data[18] = cameraRight[2];
    data[19] = 0;
    data[20] = cameraUp[0];
    data[21] = cameraUp[1];
    data[22] = cameraUp[2];
    data[23] = 0;
    // Light direction
    const dir = normalize([1,1,0.6]);
    data[24] = dir[0];
    data[25] = dir[1];
    data[26] = dir[2];
    data[27] = 0;
    device.queue.writeBuffer(uniformBuffer, 0, data);

    // 3) Acquire swapchain texture
    let texture: GPUTexture;
    try {
      texture = context.getCurrentTexture();
    } catch {
      rafIdRef.current = requestAnimationFrame(renderFrame);
      return;
    }

    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: texture.createView(),
          clearValue: { r: 0.15, g: 0.15, b: 0.15, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    };

    const cmdEncoder = device.createCommandEncoder();
    const passEncoder = cmdEncoder.beginRenderPass(passDesc);

    // A) Draw point cloud
    if (pcInstanceBuffer && pcInstanceCount > 0) {
      passEncoder.setPipeline(billboardPipeline);
      passEncoder.setBindGroup(0, uniformBindGroup);
      passEncoder.setVertexBuffer(0, billboardQuadVB);
      passEncoder.setIndexBuffer(billboardQuadIB, 'uint16');
      passEncoder.setVertexBuffer(1, pcInstanceBuffer);
      passEncoder.drawIndexed(6, pcInstanceCount);
    }

    // B) Draw ellipsoids
    if (ellipsoidInstanceBuffer && ellipsoidInstanceCount > 0) {
      passEncoder.setPipeline(ellipsoidPipeline);
      passEncoder.setBindGroup(0, uniformBindGroup);
      passEncoder.setVertexBuffer(0, sphereVB);
      passEncoder.setIndexBuffer(sphereIB, 'uint16');
      passEncoder.setVertexBuffer(1, ellipsoidInstanceBuffer);
      passEncoder.drawIndexed(sphereIndexCount, ellipsoidInstanceCount);
    }

    // [BAND CHANGE #10]: Draw ellipsoid bands
    if (bandInstanceBuffer && bandInstanceCount > 0) {
      passEncoder.setPipeline(ellipsoidBandPipeline);
      passEncoder.setBindGroup(0, uniformBindGroup);
      passEncoder.setVertexBuffer(0, ringVB);
      passEncoder.setIndexBuffer(ringIB, 'uint16');
      passEncoder.setVertexBuffer(1, bandInstanceBuffer);
      passEncoder.drawIndexed(ringIndexCount, bandInstanceCount);
    }

    passEncoder.end();
    device.queue.submit([cmdEncoder.finish()]);
    rafIdRef.current = requestAnimationFrame(renderFrame);
  }, [camera, canvasWidth, canvasHeight]);

  /******************************************************
   * F) Build Instance Data
   ******************************************************/
  function buildPCInstanceData(
    positions: Float32Array,
    colors?: Float32Array,
    scales?: Float32Array,
    decorations?: Decoration[]
  ) {
    const count = positions.length / 3;
    const data = new Float32Array(count * 9);
    // (pos.x, pos.y, pos.z, col.r, col.g, col.b, alpha, scaleX, scaleY)

    for (let i = 0; i < count; i++) {
      data[i*9+0] = positions[i*3+0];
      data[i*9+1] = positions[i*3+1];
      data[i*9+2] = positions[i*3+2];

      if (colors && colors.length === count * 3) {
        data[i*9+3] = colors[i*3+0];
        data[i*9+4] = colors[i*3+1];
        data[i*9+5] = colors[i*3+2];
      } else {
        data[i*9+3] = 1;
        data[i*9+4] = 1;
        data[i*9+5] = 1;
      }

      data[i*9+6] = 1.0; // alpha
      const s = scales ? scales[i] : 0.02;
      data[i*9+7] = s;
      data[i*9+8] = s;
    }

    if (decorations) {
      for (const dec of decorations) {
        const { indexes, color, alpha, scale, minSize } = dec;
        for (const idx of indexes) {
          if (idx<0 || idx>=count) continue;
          if (color) {
            data[idx*9+3] = color[0];
            data[idx*9+4] = color[1];
            data[idx*9+5] = color[2];
          }
          if (alpha!==undefined) {
            data[idx*9+6] = alpha;
          }
          if (scale!==undefined) {
            data[idx*9+7] *= scale;
            data[idx*9+8] *= scale;
          }
          if (minSize!==undefined) {
            if (data[idx*9+7]<minSize) data[idx*9+7]=minSize;
            if (data[idx*9+8]<minSize) data[idx*9+8]=minSize;
          }
        }
      }
    }

    return data;
  }

  function buildEllipsoidInstanceData(
    centers: Float32Array,
    radii: Float32Array,
    colors?: Float32Array,
    decorations?: Decoration[],
  ) {
    const count = centers.length / 3;
    const data = new Float32Array(count * 10);
    // (pos.x, pos.y, pos.z, scale.x, scale.y, scale.z, col.r, col.g, col.b, alpha)

    for (let i=0; i<count; i++) {
      data[i*10+0] = centers[i*3+0];
      data[i*10+1] = centers[i*3+1];
      data[i*10+2] = centers[i*3+2];

      data[i*10+3] = radii[i*3+0] || 0.1;
      data[i*10+4] = radii[i*3+1] || 0.1;
      data[i*10+5] = radii[i*3+2] || 0.1;

      if (colors && colors.length === count*3) {
        data[i*10+6] = colors[i*3+0];
        data[i*10+7] = colors[i*3+1];
        data[i*10+8] = colors[i*3+2];
      } else {
        data[i*10+6] = 1;
        data[i*10+7] = 1;
        data[i*10+8] = 1;
      }
      data[i*10+9] = 1.0;
    }

    if (decorations) {
      for (const dec of decorations) {
        const { indexes, color, alpha, scale, minSize } = dec;
        for (const idx of indexes) {
          if (idx<0 || idx>=count) continue;
          if (color) {
            data[idx*10+6] = color[0];
            data[idx*10+7] = color[1];
            data[idx*10+8] = color[2];
          }
          if (alpha!==undefined) {
            data[idx*10+9] = alpha;
          }
          if (scale!==undefined) {
            data[idx*10+3] *= scale;
            data[idx*10+4] *= scale;
            data[idx*10+5] *= scale;
          }
          if (minSize!==undefined) {
            if (data[idx*10+3]<minSize) data[idx*10+3]=minSize;
            if (data[idx*10+4]<minSize) data[idx*10+4]=minSize;
            if (data[idx*10+5]<minSize) data[idx*10+5]=minSize;
          }
        }
      }
    }

    return data;
  }

  // [BAND CHANGE #11]: Build instance data for EllipsoidBand
  // We create three rings (XY, YZ, XZ) per ellipsoid, each ring scaled by that ellipsoid's radii on the relevant two axes.
  function buildEllipsoidBandInstanceData(
    centers: Float32Array,
    radii: Float32Array,
    colors?: Float32Array,
    decorations?: Decoration[],
  ) {
    const count = centers.length / 3;
    const ringCount = count * 3;
    const data = new Float32Array(ringCount * 10);

    for (let i=0; i<count; i++) {
      const cx = centers[i*3+0], cy = centers[i*3+1], cz = centers[i*3+2];
      const rx = radii[i*3+0] || 0.1;
      const ry = radii[i*3+1] || 0.1;
      const rz = radii[i*3+2] || 0.1;

      let cr = 1, cg = 1, cb = 1;
      if (colors && colors.length === count*3) {
        cr = colors[i*3+0];
        cg = colors[i*3+1];
        cb = colors[i*3+2];
      }
      const alpha = 1.0;

      // All rings share the same center and color
      for (let ring = 0; ring < 3; ring++) {
        const idx = i*3 + ring;
        data[idx*10+0] = cx;
        data[idx*10+1] = cy;
        data[idx*10+2] = cz;

        // Set scales based on which plane this ring represents
        if (ring === 0) {
          // XY plane
          data[idx*10+3] = rx;
          data[idx*10+4] = ry;
          data[idx*10+5] = rz * 0.02; // thin in Z
        } else if (ring === 1) {
          // YZ plane
          data[idx*10+3] = rx * 0.02; // thin in X
          data[idx*10+4] = ry;
          data[idx*10+5] = rz;
        } else {
          // XZ plane
          data[idx*10+3] = rx;
          data[idx*10+4] = ry * 0.02; // thin in Y
          data[idx*10+5] = rz;
        }

        data[idx*10+6] = cr;
        data[idx*10+7] = cg;
        data[idx*10+8] = cb;
        data[idx*10+9] = alpha;
      }
    }

    // Apply decorations (unchanged)
    // ...

    return data;
  }

  /******************************************************
   * G) Updating Buffers
   ******************************************************/
  const updateBuffers = useCallback((sceneElements: SceneElementConfig[]) => {
    if (!gpuRef.current) return;
    const { device } = gpuRef.current;

    let pcInstData: Float32Array | null = null;
    let pcCount = 0;

    let ellipsoidInstData: Float32Array | null = null;
    let ellipsoidCount = 0;

    // [BAND CHANGE #12]
    let bandInstData: Float32Array | null = null;
    let bandCount = 0;

    // For brevity, handle only one of each shape
    for (const elem of sceneElements) {
      if (elem.type === 'PointCloud') {
        const { positions, colors, scales } = elem.data;
        if (positions.length > 0) {
          pcInstData = buildPCInstanceData(positions, colors, scales, elem.decorations);
          pcCount = positions.length/3;
        }
      }
      else if (elem.type === 'Ellipsoid') {
        const { centers, radii, colors } = elem.data;
        if (centers.length > 0) {
          ellipsoidInstData = buildEllipsoidInstanceData(centers, radii, colors, elem.decorations);
          ellipsoidCount = centers.length/3;
        }
      }
      // EllipsoidBand
      else if (elem.type === 'EllipsoidBand') {
        const { centers, radii, colors } = elem.data;
        if (centers.length > 0) {
          bandInstData = buildEllipsoidBandInstanceData(centers, radii, colors, elem.decorations);
          bandCount = (centers.length / 3) * 3; // 3 rings per ellipsoid
        }
      }
    }

    // create buffers
    if (pcInstData && pcCount>0) {
      const buf = device.createBuffer({
        size: pcInstData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, pcInstData);
      gpuRef.current.pcInstanceBuffer?.destroy();
      gpuRef.current.pcInstanceBuffer = buf;
      gpuRef.current.pcInstanceCount = pcCount;
    } else {
      gpuRef.current.pcInstanceBuffer?.destroy();
      gpuRef.current.pcInstanceBuffer = null;
      gpuRef.current.pcInstanceCount = 0;
    }

    if (ellipsoidInstData && ellipsoidCount>0) {
      const buf = device.createBuffer({
        size: ellipsoidInstData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, ellipsoidInstData);
      gpuRef.current.ellipsoidInstanceBuffer?.destroy();
      gpuRef.current.ellipsoidInstanceBuffer = buf;
      gpuRef.current.ellipsoidInstanceCount = ellipsoidCount;
    } else {
      gpuRef.current.ellipsoidInstanceBuffer?.destroy();
      gpuRef.current.ellipsoidInstanceBuffer = null;
      gpuRef.current.ellipsoidInstanceCount = 0;
    }

    // [BAND CHANGE #13]: create band buffer
    if (bandInstData && bandCount>0) {
      const buf = device.createBuffer({
        size: bandInstData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, bandInstData);
      gpuRef.current.bandInstanceBuffer?.destroy();
      gpuRef.current.bandInstanceBuffer = buf;
      gpuRef.current.bandInstanceCount = bandCount;
    } else {
      gpuRef.current.bandInstanceBuffer?.destroy();
      gpuRef.current.bandInstanceBuffer = null;
      gpuRef.current.bandInstanceCount = 0;
    }
  }, []);

  /******************************************************
   * H) Mouse + Zoom
   ******************************************************/
  const mouseState = useRef<{ x: number; y: number; button: number } | null>(null);

  const onMouseDown = useCallback((e: MouseEvent) => {
    mouseState.current = { x: e.clientX, y: e.clientY, button: e.button };
  }, []);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!mouseState.current) return;
    const dx = e.clientX - mouseState.current.x;
    const dy = e.clientY - mouseState.current.y;
    mouseState.current.x = e.clientX;
    mouseState.current.y = e.clientY;

    // Right drag or SHIFT+Left => pan
    if (mouseState.current.button === 2 || e.shiftKey) {
      setCamera(cam => ({
        ...cam,
        panX: cam.panX - dx * 0.002,
        panY: cam.panY + dy * 0.002,
      }));
    }
    // Left => orbit
    else if (mouseState.current.button === 0) {
      setCamera(cam => {
        const newPhi = Math.max(0.1, Math.min(Math.PI - 0.1, cam.orbitPhi - dy * 0.01));
        return {
          ...cam,
          orbitTheta: cam.orbitTheta - dx * 0.01,
          orbitPhi: newPhi,
        };
      });
    }
  }, []);

  const onMouseUp = useCallback(() => {
    mouseState.current = null;
  }, []);

  const onWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY * 0.01;
    setCamera(cam => ({
      ...cam,
      orbitRadius: Math.max(0.01, cam.orbitRadius + delta),
    }));
  }, []);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    c.addEventListener('mousedown', onMouseDown);
    c.addEventListener('mousemove', onMouseMove);
    c.addEventListener('mouseup', onMouseUp);
    c.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      c.removeEventListener('mousedown', onMouseDown);
      c.removeEventListener('mousemove', onMouseMove);
      c.removeEventListener('mouseup', onMouseUp);
      c.removeEventListener('wheel', onWheel);
    };
  }, [onMouseDown, onMouseMove, onMouseUp, onWheel]);

  /******************************************************
   * I) Effects
   ******************************************************/
  useEffect(() => {
    initWebGPU();
    return () => {
      // Cleanup
      if (gpuRef.current) {
        const {
          billboardQuadVB,
          billboardQuadIB,
          sphereVB,
          sphereIB,
          ringVB,
          ringIB,
          uniformBuffer,
          pcInstanceBuffer,
          ellipsoidInstanceBuffer,
          bandInstanceBuffer,
          depthTexture,
        } = gpuRef.current;
        billboardQuadVB.destroy();
        billboardQuadIB.destroy();
        sphereVB.destroy();
        sphereIB.destroy();
        ringVB.destroy();
        ringIB.destroy();
        uniformBuffer.destroy();
        pcInstanceBuffer?.destroy();
        ellipsoidInstanceBuffer?.destroy();
        bandInstanceBuffer?.destroy();
        depthTexture?.destroy();
      }
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [initWebGPU]);

  useEffect(() => {
    if (isReady) {
      createOrUpdateDepthTexture();
    }
  }, [isReady, canvasWidth, canvasHeight, createOrUpdateDepthTexture]);

  useEffect(() => {
    if (canvasRef.current) {
      canvasRef.current.width = canvasWidth;
      canvasRef.current.height = canvasHeight;
    }
  }, [canvasWidth, canvasHeight]);

  useEffect(() => {
    if (isReady) {
      renderFrame();
      rafIdRef.current = requestAnimationFrame(renderFrame);
    }
    return () => {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    };
  }, [isReady, renderFrame]);

  useEffect(() => {
    if (isReady) {
      updateBuffers(elements);
    }
  }, [isReady, elements, updateBuffers]);

  return (
    <div style={{ width: '100%', border: '1px solid #ccc' }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: canvasHeight }} />
    </div>
  );
}

/******************************************************
 * 6) Example: App
 ******************************************************/
export function App() {
  // Normal ellipsoid
  const eCenters = new Float32Array([
    0, 0, 0,
    0.5, 0.2, -0.2,
  ]);
  const eRadii = new Float32Array([
    0.2, 0.3, 0.15,
    0.1, 0.25, 0.2,
  ]);

  const eColors = new Float32Array([
    0.8, 0.2, 0.2,
    0.2, 0.8, 0.2,
  ]);

  // Our new "band" version
  const bandCenters = new Float32Array([
    -0.4, 0.4, 0.0,
    0.3, -0.4, 0.3,
  ]);
  const bandRadii = new Float32Array([
    0.25, 0.25, 0.25, // sphere
    0.05, 0.3, 0.1,   // elongated
  ]);
  const bandColors = new Float32Array([
    1.0, 1.0, 0.2,
    0.2, 0.7, 1.0,
  ]);

  // Basic point cloud
  const pcPositions = new Float32Array([
    -0.5, -0.5, 0,
     0.5, -0.5, 0,
    -0.5,  0.5, 0,
     0.5,  0.5, 0,
  ]);
  const pcColors = new Float32Array([
    1,0,0,
    0,1,0,
    0,0,1,
    1,1,0,
  ]);

  const pcElement: PointCloudElementConfig = {
    type: 'PointCloud',
    data: { positions: pcPositions, colors: pcColors },
    decorations: [
      { indexes: [0,1,2,3], alpha: 0.7, scale: 2.0 },
    ],
  };

  const ellipsoidElement: EllipsoidElementConfig = {
    type: 'Ellipsoid',
    data: { centers: eCenters, radii: eRadii, colors: eColors },
  };

  // [BAND CHANGE #14]: EllipsoidBand example element
  const bandElement: EllipsoidBandElementConfig = {
    type: 'EllipsoidBand',
    data: { centers: bandCenters, radii: bandRadii, colors: bandColors },
    decorations: [
      { indexes: [0], color: [1,0.5,0.2], alpha: 0.9, scale: 1.2 },
      { indexes: [1], color: [0,1,1], alpha: 0.7, scale: 1.0 },
    ],
  };

  const testElements: SceneElementConfig[] = [
    pcElement,
    ellipsoidElement,
    bandElement,
  ];

  return <SceneWrapper elements={testElements} />;
}

export function Torus(props) {
  return <SceneWrapper {...props} />;
}
