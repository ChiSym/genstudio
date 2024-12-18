import { vec3 } from 'gl-matrix';

export interface PointCloudData {
    xyz: Float32Array;
    rgb?: Uint8Array;
}

export interface CameraParams {
    position: vec3 | [number, number, number];
    target: vec3 | [number, number, number];
    up: vec3 | [number, number, number];
    fov: number;
    near: number;
    far: number;
}

export interface DecorationGroup {
  indexes: number[];
  color?: [number, number, number];
  scale?: number;
  alpha?: number;
  minSize?: number;
}

export interface DecorationGroups {
    [name: string]: DecorationGroup;
}

export interface PointCloudViewerProps {
    // Data
    points: PointCloudData;

    // Camera control
    camera?: CameraParams;
    defaultCamera?: CameraParams;
    onCameraChange?: (camera: CameraParams) => void;

    // Appearance
    backgroundColor?: [number, number, number];
    className?: string;
    pointSize?: number;

    // Dimensions
    width?: number;
    height?: number;
    aspectRatio?: number;

    // Interaction
    onPointClick?: (pointIndex: number, event: MouseEvent) => void;
    onPointHover?: (pointIndex: number | null) => void;
    pickingRadius?: number;
    decorations?: DecorationGroups;
}

export interface ShaderUniforms {
    projection: WebGLUniformLocation | null;
    view: WebGLUniformLocation | null;
    pointSize: WebGLUniformLocation | null;
    canvasSize: WebGLUniformLocation | null;
    decorationScales: WebGLUniformLocation | null;
    decorationColors: WebGLUniformLocation | null;
    decorationAlphas: WebGLUniformLocation | null;
    decorationMap: WebGLUniformLocation | null;
    decorationMapSize: WebGLUniformLocation | null;
    decorationMinSizes: WebGLUniformLocation | null;
    renderMode: WebGLUniformLocation | null;
}

export interface PickingUniforms {
    projection: WebGLUniformLocation | null;
    view: WebGLUniformLocation | null;
    pointSize: WebGLUniformLocation | null;
    canvasSize: WebGLUniformLocation | null;
}
