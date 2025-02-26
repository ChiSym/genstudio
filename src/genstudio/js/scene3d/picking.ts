export function packID(instanceIdx: number): number {
    return 1 + instanceIdx;
}

export function unpackID(id: number): number | null {
    if (id === 0) return null;
    return id - 1;
}
